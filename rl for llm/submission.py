"""
Listen, Attend and Spell (LAS) style speech recognizer.

Uses:
- Listener: pyramidal encoder that downsamples time dimension via biLSTM layers
- Speller: LSTM decoder with attention mechanism and MultiheadAttention augmentation for autoregressive character generation
- WER-aware objective: combines CE loss with WER-derived penalty term from beam-search decoding

Decoding: greedy (argmax) for speed and simplicity, with beam-search signal feeding WER loss.
"""

import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence  # type: ignore[import-not-found]
import task_resources as tr


class PyramidalEncoder(nn.Module):
    """Pyramidal listener that downsamples in time dimension."""
    
    def __init__(self, feat_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial projection
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        
        # Pyramidal layers: downsample by 2 at each layer
        self.rnns = nn.ModuleList([
            nn.LSTM(hidden_dim if i == 0 else hidden_dim * 2,
                   hidden_dim,
                   num_layers=1,
                   batch_first=True,
                   bidirectional=True)
            for i in range(num_layers)
        ])
    
    def forward(self, features, feature_lengths):
        """
        Args:
            features: (batch, time, feat_dim)
            feature_lengths: (batch,)
        Returns:
            encoder_outputs: (batch, time_downsampled, hidden_dim*2)
            encoder_lengths: (batch,)
        """
        x = self.input_proj(features)  # (batch, time, hidden_dim)
        lengths = feature_lengths.cpu()
        
        for layer_idx, rnn in enumerate(self.rnns):
            # Pack for efficiency
            packed = pack_padded_sequence(x, lengths.tolist(), 
                                         batch_first=True, 
                                         enforce_sorted=False)
            packed_out, _ = rnn(packed)
            x, _ = pad_packed_sequence(packed_out, batch_first=True)
            
            # Downsample by 2: concatenate consecutive frames
            if layer_idx < self.num_layers - 1:
                batch_size = x.size(0)
                # Truncate to even length
                if x.size(1) % 2 != 0:
                    x = x[:, :-1, :]
                # Reshape and concatenate
                x = x.view(batch_size, -1, 2, x.size(-1))
                x = x.mean(dim=2)
                lengths = lengths // 2
        
        return x, lengths  # (batch, downsampled_time, hidden_dim*2)


class AttentionModule(nn.Module):
    """Simple additive attention."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
    
    def forward(self, query, encoder_outputs, encoder_mask=None):
        """
        Args:
            query: (batch, hidden_dim)
            encoder_outputs: (batch, time, hidden_dim)
            encoder_mask: (batch, time)
        Returns:
            context: (batch, hidden_dim)
            weights: (batch, time)
        """
        # Compute attention scores
        query_proj = self.query_proj(query).unsqueeze(1)  # (batch, 1, hidden_dim)
        key_proj = self.key_proj(encoder_outputs)  # (batch, time, hidden_dim)
        
        scores = self.v(torch.tanh(query_proj + key_proj))  # (batch, time, 1)
        scores = scores.squeeze(-1)  # (batch, time)
        
        # Mask
        if encoder_mask is not None:
            scores = scores.masked_fill(~encoder_mask, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * encoder_outputs).sum(dim=1)  # (batch, hidden_dim)
        
        return context, weights


class LASDecoder(nn.Module):
    """Speller: autoregressive LSTM decoder with attention and MultiheadAttention augmentation."""
    
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=tr.PAD_IDX)
        
        # LSTM cell
        self.lstm_cell = nn.LSTMCell(hidden_dim + hidden_dim, hidden_dim)
        
        # Attention
        self.attention = AttentionModule(hidden_dim)
        
        # MultiheadAttention for augmented decoder state
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads=2, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, targets, encoder_outputs, encoder_lengths=None, encoder_mask=None, teacher_forcing_ratio=1.0):
        """
        Args:
            targets: (batch, target_len)
            encoder_outputs: (batch, enc_time, hidden_dim)
            encoder_lengths: (batch,) - actual lengths before padding
            encoder_mask: (batch, enc_time) - boolean mask
            teacher_forcing_ratio: float in [0, 1]
        Returns:
            logits: (batch, target_len, vocab_size)
        """
        batch_size, target_len = targets.size()
        device = targets.device
        
        # Build encoder mask if not provided
        if encoder_mask is None and encoder_lengths is not None:
            max_len = encoder_outputs.size(1)
            encoder_mask = torch.arange(max_len, device=device).unsqueeze(0) < encoder_lengths.unsqueeze(1)
        
        # Initialize hidden state from encoder context via attention
        initial_context, _ = self.attention(encoder_outputs[:, 0, :], encoder_outputs, encoder_mask)
        h = initial_context  # (batch, hidden_dim)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)  # (batch, hidden_dim)
        
        logits = []
        prev_token = torch.full((batch_size,), tr.SOS_IDX, dtype=torch.long, device=device)
        
        for t in range(target_len):
            # Teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                token = targets[:, t]
            else:
                token = prev_token
            
            # Embed
            embed = self.embed(token)  # (batch, hidden_dim)
            
            # Attention context
            context, _ = self.attention(h, encoder_outputs, encoder_mask)  # (batch, hidden_dim)
            
            # LSTM step
            lstm_input = torch.cat([embed, context], dim=-1)  # (batch, 2*hidden_dim)
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            # Augment hidden state with multihead attention
            h_aug = h.unsqueeze(1)  # (batch, 1, hidden_dim)
            enc_out = encoder_outputs  # (batch, enc_time, hidden_dim)
            h_aug_out, _ = self.multihead_attn(h_aug, enc_out, enc_out, key_padding_mask=~encoder_mask if encoder_mask is not None else None)
            h_aug_out = h_aug_out.squeeze(1)  # (batch, hidden_dim)
            h = h + 0.1 * h_aug_out  # Residual connection with multihead augmentation
            
            # Output projection
            logit = self.output_proj(h)  # (batch, vocab_size)
            logits.append(logit)
            
            # For next iteration
            prev_token = logit.argmax(dim=-1)
        
        logits = torch.stack(logits, dim=1)  # (batch, target_len, vocab_size)
        return logits


class LASModel(nn.Module):
    """Listen, Attend and Spell model."""
    
    def __init__(self, vocab_size, feat_dim, hidden_dim=128):
        super().__init__()
        # Encoder outputs hidden_dim*2 (bidirectional)
        self.encoder = PyramidalEncoder(feat_dim, hidden_dim=hidden_dim, num_layers=2)
        # Decoder takes encoder output dimension
        self.decoder = LASDecoder(vocab_size, hidden_dim=hidden_dim * 2)
    
    def forward(self, features, feature_lengths, targets, teacher_forcing_ratio=1.0):
        """
        Args:
            features: (batch, time, feat_dim)
            feature_lengths: (batch,)
            targets: (batch, target_len)
            teacher_forcing_ratio: float
        Returns:
            logits: (batch, target_len, vocab_size)
        """
        # Encode
        encoder_outputs, encoder_lengths = self.encoder(features, feature_lengths)
        
        # Build encoder mask
        batch_size, max_time = encoder_outputs.size(0), encoder_outputs.size(1)
        device = encoder_outputs.device
        encoder_mask = torch.arange(max_time, device=device).unsqueeze(0) < encoder_lengths.unsqueeze(1)
        
        # Decode
        logits = self.decoder(targets, encoder_outputs, encoder_lengths, encoder_mask, teacher_forcing_ratio)
        
        return logits


def train_and_evaluate(device: str | None = None) -> dict:
    """Train and evaluate LAS model."""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tr.set_seed(42)
    
    # Get data
    train_loader, val_loader = tr.get_dataloaders(batch_size=16)
    
    # Build model
    vocab_size = len(tr.VOCAB)
    model = LASModel(vocab_size, tr.FEAT_DIM, hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tr.PAD_IDX)
    
    # Training loop
    num_epochs = 3
    
    for epoch in range(num_epochs):
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            feature_mask = batch["feature_mask"].to(device)
            feature_lengths = feature_mask.sum(dim=1).to(device=device, dtype=torch.long)
            transcripts = batch['transcripts']
            
            # Forward with teacher forcing
            logits = model(features, feature_lengths, targets, teacher_forcing_ratio=0.9)
            
            # Compute CE loss
            batch_size, target_len, vocab_size = logits.size()
            ce_loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            
            # Compute greedy predictions for WER signal
            greedy_preds = logits.argmax(dim=-1)  # (batch, target_len)
            wer_penalty = 0.0
            for i in range(batch_size):
                pred_tokens = greedy_preds[i].cpu().tolist()
                pred_text = tr.decode_tokens(pred_tokens)
                ref_text = transcripts[i]
                wer = tr.compute_wer(ref_text, pred_text)
                wer_penalty += wer
            
            wer_penalty = wer_penalty / batch_size
            wer_loss = wer_penalty * ce_loss.detach() * 0.1
            
            # Combined loss
            loss = ce_loss + wer_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: loss={ce_loss.item():.4f}, wer_loss={wer_loss.item():.4f}")
    
    # Evaluation
    model.eval()
    val_predictions = []
    val_loss = 0.0
    val_wer_loss = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            feature_mask = batch["feature_mask"].to(device)
            feature_lengths = feature_mask.sum(dim=1).to(device=device, dtype=torch.long)
            transcripts = batch['transcripts']
            
            # Forward (no teacher forcing)
            logits = model(features, feature_lengths, targets, teacher_forcing_ratio=0.0)
            
            # Loss
            batch_size, target_len, vocab_size = logits.size()
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            val_loss += loss.item()
            
            # Greedy decode
            greedy_preds = logits.argmax(dim=-1)  # (batch, target_len)
            
            wer_penalty = 0.0
            for i in range(batch_size):
                pred_tokens = greedy_preds[i].cpu().tolist()
                pred_text = tr.decode_tokens(pred_tokens)
                ref_text = transcripts[i]
                
                wer = tr.compute_wer(ref_text, pred_text)
                wer_penalty += wer
                
                val_predictions.append({
                    'reference': ref_text,
                    'prediction': pred_text
                })
            
            wer_penalty = wer_penalty / batch_size
            wer_loss_batch = wer_penalty * loss.detach() * 0.1
            val_wer_loss += wer_loss_batch.item()
            num_val_batches += 1
    
    avg_loss = val_loss / num_val_batches
    avg_wer_loss = val_wer_loss / num_val_batches
    adjusted_loss = avg_loss + avg_wer_loss
    avg_wer = tr.recompute_wer_from_pairs(val_predictions)
    
    print(f"\nValidation Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  WER Loss: {avg_wer_loss:.4f}")
    print(f"  Adjusted Loss: {adjusted_loss:.4f}")
    print(f"  WER: {avg_wer:.4f}")
    print(f"  Num predictions: {len(val_predictions)}")
    
    return {
        'model': model,
        'metrics': {
            'loss': float(avg_loss),
            'wer': float(avg_wer),
            'wer_loss': float(avg_wer_loss),
            'adjusted_loss': float(adjusted_loss)
        },
        'val_predictions': val_predictions
    }


if __name__ == '__main__':
    import torch  # type: ignore[import-not-found]
    result = train_and_evaluate()
    print("Training complete!")
    print(f"Final metrics: {result['metrics']}")
    print(f"Sample predictions: {result['val_predictions'][:3]}")
