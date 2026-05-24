import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
from src.config import EMBEDDING_DIM, MMOE_NUM_EXPERTS, MMOE_EXPERT_HIDDEN, MMOE_TOWER_HIDDEN
from src.utils.logger import logger

@dataclass
class ObjectiveWeights:
    """
    Standard objective weights for blended multi-task ranking utility score.
    Capped clickbait CTR bias by weighting watch completion and likes higher, 
    and penalizing explicit dislike actions.
    """
    click: float = 0.20
    watch_complete: float = 0.35  # most important engagement indicator
    like: float = 0.20
    share: float = 0.15
    dislike_penalty: float = -0.10
    diversity_bonus: float = 0.10

class ExpertNetwork(nn.Module):
    """
    Shared Expert network representation block.
    Small MLP (input -> 64 -> 64) with LayerNorm and ReLU.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(ExpertNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class GatingNetwork(nn.Module):
    """
    Softmax-routing Gating Network for a specific task.
    Linear(input -> num_experts) + Softmax.
    """
    def __init__(self, input_dim: int, num_experts: int):
        super(GatingNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return self.softmax(logits)

class TaskTower(nn.Module):
    """
    Task-specific output Tower.
    Small MLP (64 -> 32 -> 1) + Sigmoid.
    """
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32):
        super(TaskTower, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)

class MultiObjectiveRankingModel(nn.Module):
    """
    Multi-gate Mixture-of-Experts (MMoE) model for Multi-Objective YouTube ranking.
    Features 3 shared experts, 4 task gates, and 4 task towers.
    """
    def __init__(self, input_dim: int, num_experts: int = 3, expert_hidden: int = 64):
        super(MultiObjectiveRankingModel, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        
        # 1. Shared Expert Networks
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, expert_hidden) for _ in range(num_experts)
        ])
        
        # 2. 4 Task Gates (one per task: click, watch_complete, like, dislike)
        self.gates = nn.ModuleDict({
            'click': GatingNetwork(input_dim, num_experts),
            'watch_complete': GatingNetwork(input_dim, num_experts),
            'like': GatingNetwork(input_dim, num_experts),
            'dislike': GatingNetwork(input_dim, num_experts)
        })
        
        # 3. 4 Task Towers (Sigmoid mapping to probability)
        self.towers = nn.ModuleDict({
            'click': TaskTower(expert_hidden),
            'watch_complete': TaskTower(expert_hidden),
            'like': TaskTower(expert_hidden),
            'dislike': TaskTower(expert_hidden)
        })
        
        # Load standard objective weights definition
        self.weights = ObjectiveWeights()
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Evaluate all experts
        # expert_outputs shape: [Num_Experts, Batch_Size, Expert_Hidden]
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        
        outputs = {}
        for task in ['click', 'watch_complete', 'like', 'dislike']:
            # Gate weights for current task: [Batch_Size, Num_Experts]
            gate_weights = self.gates[task](x)
            
            # Weighted expert mix: [Batch_Size, Expert_Hidden]
            weighted_experts = expert_outputs.transpose(0, 1) * gate_weights.unsqueeze(-1)
            blended_rep = torch.sum(weighted_experts, dim=1)
            
            # Tower prediction [Batch_Size]
            outputs[task] = self.towers[task](blended_rep)
            
        return outputs

    def get_final_score(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Blended Utility Score combining all task objective weights.
        """
        score = (
            self.weights.click * outputs['click'] +
            self.weights.watch_complete * outputs['watch_complete'] +
            self.weights.like * outputs['like'] +
            self.weights.dislike_penalty * outputs['dislike']
        )
        return score

    def compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Weighted BCE Loss across all tasks.
        watch_complete gets 2x weight multiplier!
        """
        loss_fn = nn.BCELoss()
        
        loss_click = loss_fn(outputs['click'], targets['click'])
        loss_watch = loss_fn(outputs['watch_complete'], targets['watch_complete'])
        loss_like = loss_fn(outputs['like'], targets['like'])
        loss_dislike = loss_fn(outputs['dislike'], targets['dislike'])
        
        # watch_complete is scaled by 2x to force system optimization on long engagement
        total_loss = loss_click + 2.0 * loss_watch + loss_like + loss_dislike
        
        losses = {
            'total': float(total_loss.item()),
            'click': float(loss_click.item()),
            'watch_complete': float(loss_watch.item()),
            'like': float(loss_like.item()),
            'dislike': float(loss_dislike.item())
        }
        return total_loss, losses


# =====================================================================
# BACKWARD COMPATIBLE ORIGINAL SINGLE-GATE EXPERT CLASSES
# =====================================================================

class MMoEModel(nn.Module):
    """
    Legacy two-objective model kept to maintain perfect API compatibility.
    """
    def __init__(self, input_dim: int, num_experts: int = MMOE_NUM_EXPERTS, 
                 expert_hidden: int = MMOE_EXPERT_HIDDEN, tower_hidden: int = MMOE_TOWER_HIDDEN):
        super(MMoEModel, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden, expert_hidden),
                nn.ReLU()
            ) for _ in range(num_experts)
        ])
        
        self.gate_click = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
        self.gate_watch = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
        self.tower_click = nn.Sequential(
            nn.Linear(expert_hidden, tower_hidden),
            nn.ReLU(),
            nn.Linear(tower_hidden, 1),
            nn.Sigmoid()
        )
        
        self.tower_watch = nn.Sequential(
            nn.Linear(expert_hidden, tower_hidden),
            nn.ReLU(),
            nn.Linear(tower_hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        
        gate_click_weights = self.gate_click(x)
        weighted_expert_click = expert_outputs.transpose(0, 1) * gate_click_weights.unsqueeze(-1)
        blended_click = torch.sum(weighted_expert_click, dim=1)
        pred_click = self.tower_click(blended_click).squeeze(-1)
        
        gate_watch_weights = self.gate_watch(x)
        weighted_expert_watch = expert_outputs.transpose(0, 1) * gate_watch_weights.unsqueeze(-1)
        blended_watch = torch.sum(weighted_expert_watch, dim=1)
        pred_watch = self.tower_watch(blended_watch).squeeze(-1)
        
        return pred_click, pred_watch

class MMoERankingEngine:
    """
    Legacy ranking manager.
    """
    def __init__(self, input_dim: int, epochs: int = 12, lr: float = 0.002, batch_size: int = 64):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MMoEModel(input_dim=input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.click_loss_fn = nn.BCELoss()
        self.watch_loss_fn = nn.MSELoss()

    def train_model(self, X_train: np.ndarray, y_click: np.ndarray, y_watch: np.ndarray):
        logger.info(f"Training Multi-Objective MMoE on device: {self.device}. Dataset size: {len(X_train)}")
        self.model.train()
        num_samples = len(X_train)
        
        for epoch in range(self.epochs):
            shuffled_indices = np.random.permutation(num_samples)
            epoch_loss = 0.0
            
            for i in range(0, num_samples, self.batch_size):
                batch_idx = shuffled_indices[i : i + self.batch_size]
                X_batch = torch.tensor(X_train[batch_idx], dtype=torch.float32, device=self.device)
                y_click_batch = torch.tensor(y_click[batch_idx], dtype=torch.float32, device=self.device)
                y_watch_batch = torch.tensor(y_watch[batch_idx], dtype=torch.float32, device=self.device)
                
                self.optimizer.zero_grad()
                pred_click, pred_watch = self.model(X_batch)
                
                click_loss = self.click_loss_fn(pred_click, y_click_batch)
                watch_loss = self.watch_loss_fn(pred_watch, y_watch_batch)
                total_loss = click_loss + 2.0 * watch_loss
                
                total_loss.backward()
                self.optimizer.step()
                epoch_loss += total_loss.item() * len(batch_idx)
                
            avg_loss = epoch_loss / num_samples
            logger.info(f"MMoE Epoch {epoch+1}/{self.epochs} - Combined Train Loss: {avg_loss:.4f}")

    def score_candidates(self, X_candidates: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.tensor(X_candidates, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred_click, pred_watch = self.model(X_tensor)
            pred_watch = torch.clamp(pred_watch, min=0.0)
            utility_score = pred_click * pred_watch
        return utility_score.cpu().numpy()


# =====================================================================
# DRY-RUN VERIFICATION BLOCK
# =====================================================================
if __name__ == "__main__":
    import time
    logger.info("Initializing MMoE Multi-Objective Ranking Dry-Run Verification...")
    
    # 1. Generate synthetic training data
    num_samples = 1000
    input_dim = 32
    
    # Random rich feature representation
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    
    # Sigmoidal correlation formulas representing positive & negative outcomes
    y_click = (1.0 / (1.0 + np.exp(-(X[:, 0] * 1.5 + X[:, 1] * 0.8 + np.random.randn(num_samples) * 0.2)))).astype(np.float32)
    y_watch = (1.0 / (1.0 + np.exp(-(X[:, 2] * 2.0 + X[:, 3] * 1.2 + np.random.randn(num_samples) * 0.1)))).astype(np.float32)
    y_like = (1.0 / (1.0 + np.exp(-(X[:, 0] * 1.2 - X[:, 4] * 0.5 + np.random.randn(num_samples) * 0.3)))).astype(np.float32)
    y_dislike = (1.0 / (1.0 + np.exp(-(-X[:, 1] * 1.0 + X[:, 5] * 0.8 + np.random.randn(num_samples) * 0.4)))).astype(np.float32)
    
    # Convert to standard PyTorch tensors
    X_tensor = torch.tensor(X)
    targets = {
        'click': torch.tensor(y_click),
        'watch_complete': torch.tensor(y_watch),
        'like': torch.tensor(y_like),
        'dislike': torch.tensor(y_dislike)
    }
    
    # 2. Compile MultiObjectiveRankingModel
    model = MultiObjectiveRankingModel(input_dim=input_dim, num_experts=3, expert_hidden=64)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    logger.info("✅ Multi-gate Mixture-of-Experts (MMoE) model compiled successfully.")
    
    # 3. Train for 3 Epochs
    epochs = 3
    batch_size = 128
    
    logger.info("Starting MMoE multi-task optimization for 3 epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        shuffled_indices = np.random.permutation(num_samples)
        epoch_losses = {'total': 0.0, 'click': 0.0, 'watch_complete': 0.0, 'like': 0.0, 'dislike': 0.0}
        
        for i in range(0, num_samples, batch_size):
            batch_idx = shuffled_indices[i : i + batch_size]
            X_batch = X_tensor[batch_idx]
            
            targets_batch = {
                task: targets[task][batch_idx] for task in ['click', 'watch_complete', 'like', 'dislike']
            }
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss, loss_breakdown = model.compute_loss(outputs, targets_batch)
            loss.backward()
            optimizer.step()
            
            for k in epoch_losses.keys():
                epoch_losses[k] += loss_breakdown[k] * len(batch_idx)
                
        # Average epoch loss
        for k in epoch_losses.keys():
            epoch_losses[k] /= num_samples
            
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Total Loss: {epoch_losses['total']:.4f} | "
            f"Click Loss: {epoch_losses['click']:.4f} | "
            f"Watch Loss: {epoch_losses['watch_complete']:.4f} | "
            f"Like Loss: {epoch_losses['like']:.4f} | "
            f"Dislike Loss: {epoch_losses['dislike']:.4f}"
        )
        
    elapsed = time.time() - start_time
    logger.info(f"✅ MMoE multi-objective training completed in {elapsed:.2f} seconds.")
    
    # 4. Predict and evaluate utility scores
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor[:5])
        final_scores = model.get_final_score(preds)
        logger.info("Sample Candidate Predictions (First 5):")
        for i in range(5):
            logger.info(
                f"Candidate {i+1} | "
                f"Click Prob: {preds['click'][i]:.4f} | "
                f"Watch Complete: {preds['watch_complete'][i]:.4f} | "
                f"Like Prob: {preds['like'][i]:.4f} | "
                f"Dislike Prob: {preds['dislike'][i]:.4f} | "
                f"🔥 Blended Ranking Utility Score: {final_scores[i]:.4f}"
            )
            
    logger.info("All MMoE multi-objective ranking dry-run tests passed successfully!")
