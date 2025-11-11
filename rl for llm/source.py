"""
Listen, Attend and Spell (LAS) style speech recognizer.

Uses:
- Listener: pyramidal encoder that downsamples time dimension via biLSTM layers
- Speller: LSTM decoder with attention mechanism for autoregressive character generation
- WER-aware objective: combines CE loss with WER-derived penalty term

Decoding: greedy (argmax) for speed and simplicity.
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
            hidden_states: tuple of (h, c) for each direction
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
                x = x.mean(dim=2)  # or could concatenate: torch.cat on dim=-1
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
    """Speller: autoregressive LSTM decoder with attention."""
    
    def __init__(self, vocab_size, hidden_dim=128, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=tr.PAD_IDX)
        
        # LSTM cell
        self.lstm_cell = nn.LSTMCell(hidden_dim + hidden_dim, hidden_dim)
        
        # Attention
        self.attention = AttentionModule(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, targets, encoder_outputs, encoder_mask=None, teacher_forcing_ratio=1.0):
        batch_size, target_len = targets.size()
        
        # Initialize hidden state from encoder context
        initial_context, _ = self.attention(...) #complete the attention module
        # complete the forward pass
        return logits


class LASModel(nn.Module):
    """Listen, Attend and Spell model."""
    # complete the model


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
    
    # Training loop - just a few epochs
    num_epochs = 3
    accumulated_wer_loss = 0.0
    num_batches_train = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_wer_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            
            # here write train loop
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: loss={loss.item():.4f}, wer_loss={wer_loss.item():.4f}")
    
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
