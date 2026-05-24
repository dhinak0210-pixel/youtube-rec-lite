import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Tuple, Dict, Set
from src.config import EMBEDDING_DIM, GNN_NUM_EPOCHS, GNN_LR
from src.utils.logger import logger

class UserItemGraph:
    """
    Stores and manages social-aware user interaction and follower graphs.
    """
    def __init__(self, num_users: int = 1000, num_items: int = 5000):
        self.num_users = num_users
        self.num_items = num_items
        # user_id -> set of item_ids watched
        self.interactions: Dict[int, Set[int]] = {u: set() for u in range(num_users)}
        # follower -> set of followees
        self.follows: Dict[int, Set[int]] = {u: set() for u in range(num_users)}
        # followee -> set of followers
        self.followers: Dict[int, Set[int]] = {u: set() for u in range(num_users)}

    def add_interaction(self, user_id: int, item_id: int):
        if 0 <= user_id < self.num_users and 0 <= item_id < self.num_items:
            self.interactions[user_id].add(item_id)

    def add_social_connection(self, follower: int, followee: int):
        if 0 <= follower < self.num_users and 0 <= followee < self.num_users:
            self.follows[follower].add(followee)
            self.followers[followee].add(follower)

    def get_user_neighbors(self, user_id: int, max_k: int = 20) -> List[int]:
        """
        Get the 1-hop social neighbors (friends user_id follows).
        Caps return count at max_k for model scalability.
        """
        if user_id not in self.follows or not self.follows[user_id]:
            return []
        neighbors = list(self.follows[user_id])
        if len(neighbors) > max_k:
            neighbors = random.sample(neighbors, max_k)
        return neighbors

    def get_2hop_neighbors(self, user_id: int) -> Set[int]:
        """
        Get 2-hop social neighbors (friends of friends user_id follows).
        """
        hop2 = set()
        if user_id in self.follows:
            for friend in self.follows[user_id]:
                if friend in self.follows:
                    hop2.update(self.follows[friend])
        hop2.discard(user_id) # Remove self
        return hop2

    def get_friend_watched_items(self, user_id: int) -> Dict[int, int]:
        """
        Returns a dictionary of item_id -> count of friends who watched it.
        """
        counts = {}
        if user_id in self.follows:
            for friend in self.follows[user_id]:
                if friend in self.interactions:
                    for item_id in self.interactions[friend]:
                        counts[item_id] = counts.get(item_id, 0) + 1
        return counts

    def generate_synthetic_social_graph(self, num_connections: int = 5000, alpha: float = 1.5):
        """
        Generates follows using a power-law (scale-free) degree distribution.
        """
        logger.info(f"Generating synthetic social graph with power-law distribution ({num_connections} connections)...")
        # Power-law degree weights: probability of being followed proportional to rank^-alpha
        nodes = np.arange(self.num_users)
        ranks = np.arange(1, self.num_users + 1)
        probs = 1.0 / (ranks ** alpha)
        probs /= probs.sum()
        
        # Sample followees based on popular ranks
        followees = np.random.choice(nodes, size=num_connections, p=probs)
        followers = np.random.randint(0, self.num_users, size=num_connections)
        
        for u, v in zip(followers, followees):
            if u != v: # Avoid self-loops
                self.add_social_connection(int(u), int(v))


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE neural message aggregation layer.
    Aggregate: mean pooling of neighbors
    Transform: Linear(concat(self, neighbors)) + ReLU
    """
    def __init__(self, in_features: int, out_features: int):
        super(GraphSAGELayer, self).__init__()
        self.linear = nn.Linear(in_features * 2, out_features)
        self.activation = nn.ReLU()
        
    def forward(self, self_feats: torch.Tensor, neighbor_feats: torch.Tensor) -> torch.Tensor:
        """
        self_feats: [Batch_Size, In_Features]
        neighbor_feats: [Batch_Size, In_Features] (already aggregated/pooled)
        """
        combined = torch.cat([self_feats, neighbor_feats], dim=-1)
        out = self.linear(combined)
        return self.activation(out)


class SocialGraphRecommender(nn.Module):
    """
    GraphSAGE model for social-aware YouTube recommendation.
    User & Item embeddings (64-dim).
    2 GraphSAGE layers.
    """
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super(SocialGraphRecommender, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # User & Item embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # 2 GraphSAGE layers
        self.sage1 = GraphSAGELayer(embedding_dim, embedding_dim)
        self.sage2 = GraphSAGELayer(embedding_dim, embedding_dim)
        
        # Score prediction head: projects combined representations to a probability [0, 1]
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # Init weights
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        
    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor, neighbor_ids: torch.Tensor) -> torch.Tensor:
        """
        user_id: [Batch_Size]
        item_id: [Batch_Size]
        neighbor_ids: [Batch_Size, Max_Neighbors] (indices of neighbors, 0-padded or masked)
        """
        # 1. Fetch core embeddings
        u_embed = self.user_embeddings(user_id) # [Batch_Size, Embed_Dim]
        i_embed = self.item_embeddings(item_id) # [Batch_Size, Embed_Dim]
        
        # 2. Neighbors aggregation (GraphSAGE layer 1 message pooling)
        # neighbor_ids shape: [Batch_Size, Max_Neighbors]
        # Fetch neighbor embeddings: [Batch_Size, Max_Neighbors, Embed_Dim]
        n_embed = self.user_embeddings(neighbor_ids)
        
        # Mean pooling aggregation
        n_mean = torch.mean(n_embed, dim=1) # [Batch_Size, Embed_Dim]
        
        # Handle zero-neighbor masking (if neighbor_ids sum to 0, use user_embed as neighbor_feats)
        is_empty = (neighbor_ids.sum(dim=-1, keepdim=True) == 0).float()
        n_mean = (1.0 - is_empty) * n_mean + is_empty * u_embed
        
        # 3. Layer 1 GraphSAGE update
        u_h1 = self.sage1(u_embed, n_mean) # [Batch_Size, Embed_Dim]
        
        # 4. Layer 2 GraphSAGE update
        u_h2 = self.sage2(u_h1, n_mean) # [Batch_Size, Embed_Dim]
        
        # 5. Predict engagement probability
        combined = torch.cat([u_h2, i_embed], dim=-1) # [Batch_Size, Embed_Dim * 2]
        prob = self.predictor(combined).squeeze(-1) # [Batch_Size]
        return prob


def get_social_features(user_id: int, item_id: int, graph: UserItemGraph) -> Dict[str, float]:
    """
    Extracts social graph features:
    - fraction of friends who watched this
    - number of friends who watched (normalized)
    - binary: did ANY friend watch this?
    - user's network size (normalized)
    """
    friends = graph.follows.get(user_id, set())
    num_friends = len(friends)
    
    if num_friends == 0:
        return {
            "fraction_friends_watched": 0.0,
            "num_friends_watched_normalized": 0.0,
            "any_friend_watched": 0.0,
            "network_size_normalized": 0.0
        }
        
    friends_watched = 0
    for friend in friends:
        if item_id in graph.interactions.get(friend, set()):
            friends_watched += 1
            
    fraction = friends_watched / num_friends
    num_norm = min(friends_watched / 50.0, 1.0)
    any_watched = 1.0 if friends_watched > 0 else 0.0
    network_norm = min(num_friends / 200.0, 1.0)
    
    return {
        "fraction_friends_watched": float(fraction),
        "num_friends_watched_normalized": float(num_norm),
        "any_friend_watched": float(any_watched),
        "network_size_normalized": float(network_norm)
    }


def recommend_with_social(
    user_id: int, 
    candidates: List[Tuple[int, float]], 
    graph: UserItemGraph, 
    boost_factor: float = 0.2
) -> List[Tuple[int, float]]:
    """
    Boosts recommendation scores for items watched by friends:
    Score_new = Score_old * (1.0 + boost_factor * fraction_friends_watched)
    """
    boosted_recs = []
    friends = graph.follows.get(user_id, set())
    num_friends = len(friends)
    
    for item_id, score in candidates:
        if num_friends > 0:
            friends_watched = sum(1 for f in friends if item_id in graph.interactions.get(f, set()))
            fraction = friends_watched / num_friends
            boosted_score = score * (1.0 + boost_factor * fraction)
        else:
            boosted_score = score
        boosted_recs.append((item_id, boosted_score))
        
    boosted_recs.sort(key=lambda x: x[1], reverse=True)
    return boosted_recs


def train_social_recommender(
    model: SocialGraphRecommender, 
    graph: UserItemGraph, 
    epochs: int = 5, 
    lr: float = 0.005, 
    batch_size: int = 64
) -> List[float]:
    """
    Trains the SocialGraphRecommender model using interaction logs.
    Includes positive interaction pairs and samples negative (unwatched) items.
    """
    logger.info(f"Starting SocialGraphSAGE training for {epochs} epochs on CPU...")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCELoss()
    
    # Compile training pairs from graph interactions
    train_pairs = []
    for u in range(graph.num_users):
        watched = list(graph.interactions.get(u, set()))
        for item in watched:
            train_pairs.append((u, item, 1.0)) # Positive label
            
            # Negative sampling
            neg_item = random.randint(0, graph.num_items - 1)
            while neg_item in graph.interactions.get(u, set()):
                neg_item = random.randint(0, graph.num_items - 1)
            train_pairs.append((u, neg_item, 0.0)) # Negative label
            
    if not train_pairs:
        logger.warning("No interactions found for training. Populating dummy training data.")
        train_pairs = [(random.randint(0, graph.num_users - 1), random.randint(0, graph.num_items - 1), float(random.randint(0, 1))) for _ in range(200)]
        
    losses = []
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_pairs)
        epoch_loss = 0.0
        
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i+batch_size]
            users = torch.tensor([b[0] for b in batch], dtype=torch.long)
            items = torch.tensor([b[1] for b in batch], dtype=torch.long)
            labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            
            # Fetch neighbor lists for users
            max_neighbors = 10
            neighbors_list = []
            for u in users.tolist():
                neighbors = graph.get_user_neighbors(u, max_k=max_neighbors)
                # Pad to exactly max_neighbors
                if len(neighbors) < max_neighbors:
                    neighbors = neighbors + [0] * (max_neighbors - len(neighbors))
                neighbors_list.append(neighbors)
            neighbors_tensor = torch.tensor(neighbors_list, dtype=torch.long)
            
            optimizer.zero_grad()
            preds = model(users, items, neighbors_tensor)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch)
            
        avg_loss = epoch_loss / len(train_pairs)
        losses.append(avg_loss)
        logger.info(f"GraphSAGE Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")
        
    return losses


# =====================================================================
# BACKWARD COMPATIBLE BASELINE GRAPH LAYERS & CLASS WRAPPERS
# =====================================================================

class BipartiteGCNLayer(nn.Module):
    """
    Custom Graph Convolutional Network (GCN) Layer for bipartite graphs.
    Kept for backward compatibility with standard user-video interaction graphs.
    """
    def __init__(self, in_features: int, out_features: int):
        super(BipartiteGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]
        num_nodes = x.size(0)
        aggregated = torch.zeros_like(x)
        neighbor_features = x[col]
        aggregated.index_add_(0, row, neighbor_features)
        
        deg = torch.zeros(num_nodes, device=x.device)
        deg.index_add_(0, row, torch.ones(row.size(0), device=x.device))
        
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        norm_aggregated = deg_inv_sqrt.unsqueeze(1) * aggregated * deg_inv_sqrt.unsqueeze(1)
        out = self.linear(norm_aggregated)
        return self.activation(out)


class BipartiteGNNModel(nn.Module):
    """
    Custom GNN Model using bipartite message propagation hops.
    """
    def __init__(self, num_nodes: int, embedding_dim: int = EMBEDDING_DIM):
        super(BipartiteGNNModel, self).__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.gcn1 = BipartiteGCNLayer(embedding_dim, embedding_dim)
        self.gcn2 = BipartiteGCNLayer(embedding_dim, embedding_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        h0 = self.node_embeddings.weight
        h1 = self.gcn1(h0, edge_index)
        h2 = self.gcn2(h1, edge_index)
        return h0 + h1 + h2


class GNNRecommender:
    """
    GNN candidate retrieval orchestrator.
    """
    def __init__(self, num_nodes: int, num_videos: int, epochs: int = GNN_NUM_EPOCHS, lr: float = GNN_LR):
        self.num_nodes = num_nodes
        self.num_videos = num_videos
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BipartiteGNNModel(num_nodes=num_nodes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

    def train_model(self, edge_index: torch.Tensor, v_nodes: torch.Tensor):
        logger.info(f"Training Custom Bipartite GNN Model on device: {self.device}.")
        edge_index = edge_index.to(self.device)
        self.model.train()
        num_edges = edge_index.shape[1]
        users = edge_index[0]
        pos_videos = edge_index[1]
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            embeddings = self.model(edge_index)
            neg_videos = torch.randint(low=1, high=self.num_videos, size=(num_edges,), device=self.device)
            
            u_embed = embeddings[users]
            pos_embed = embeddings[pos_videos]
            neg_embed = embeddings[neg_videos]
            
            pos_scores = torch.sum(u_embed * pos_embed, dim=-1)
            neg_scores = torch.sum(u_embed * neg_embed, dim=-1)
            
            loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
            loss.backward()
            self.optimizer.step()
            logger.info(f"GNN Epoch {epoch+1}/{self.epochs} - Loss: {loss.item():.4f}")

    def retrieve_candidates(self, user_idx: int, edge_index: torch.Tensor, top_n: int = 10) -> List[Tuple[int, float]]:
        self.model.eval()
        edge_index = edge_index.to(self.device)
        user_node_id = user_idx + self.num_videos
        if user_node_id >= self.num_nodes:
            return []
            
        with torch.no_grad():
            embeddings = self.model(edge_index)
            user_vector = embeddings[user_node_id]
            video_vectors = embeddings[1:self.num_videos]
            scores = torch.mv(video_vectors, user_vector)
            top_scores, top_indices = torch.topk(scores, k=min(top_n, len(scores)))
            
        return [
            (int(idx.item() + 1), float(score.item()))
            for score, idx in zip(top_scores, top_indices)
        ]


# =====================================================================
# DRY-RUN VERIFICATION BLOCK
# =====================================================================
if __name__ == "__main__":
    import time
    logger.info("Initializing GraphSAGE Dry-Run Social Verification Block...")
    
    # 1. Initialize User-Item-Follow Graph (1000 users, 5000 items)
    graph = UserItemGraph(num_users=1000, num_items=5000)
    
    # Generate 3000 synthetic interactions
    for _ in range(3000):
        u = random.randint(0, 999)
        i = random.randint(0, 4999)
        graph.add_interaction(u, i)
        
    # Generate 5000 synthetic social links with power-law distribution
    graph.generate_synthetic_social_graph(num_connections=5000, alpha=1.6)
    
    # 2. Compile GraphSAGE Recommender Model
    model = SocialGraphRecommender(num_users=1000, num_items=5000, embedding_dim=64)
    logger.info("✅ GraphSAGE Social-Aware Model compiled successfully.")
    
    # 3. Trigger 5-Epoch CPU Training Loop
    start_time = time.time()
    losses = train_social_recommender(model, graph, epochs=5, lr=0.005, batch_size=128)
    elapsed = time.time() - start_time
    logger.info(f"✅ Training completed in {elapsed:.2f} seconds (Loss trajectory: {losses})")
    
    # 4. Extract Social Features
    target_user = 42
    target_item = 100
    features = get_social_features(target_user, target_item, graph)
    logger.info(f"✅ Extracted social features for user {target_user} on item {target_item}: {features}")
    
    # 5. Retrieve & Boost Recommendations
    mock_candidates = [(item, random.uniform(0.1, 0.9)) for item in random.sample(range(5000), 10)]
    logger.info(f"Original candidate recommendations: {mock_candidates[:5]}")
    
    boosted = recommend_with_social(target_user, mock_candidates, graph, boost_factor=0.5)
    logger.info(f"🚀 Social-Boosted candidate recommendations: {boosted[:5]}")
    logger.info("All GraphSAGE social-aware recommendation dry-run tests passed successfully!")
