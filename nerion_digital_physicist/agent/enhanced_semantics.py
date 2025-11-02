"""
Enhanced Semantic Analysis

Provides semantic embeddings and similarity analysis for architectural understanding.
Complements structural analysis with semantic meaning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class SemanticAnalyzer:
    """
    Semantic analysis for code modules.

    Uses CodeBERT embeddings to compute semantic similarity between modules.
    """

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """
        Initialize semantic analyzer.

        Args:
            model_name: Hugging Face model name for embeddings
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.embeddings_cache: Dict[str, np.ndarray] = {}

        if HAS_TRANSFORMERS:
            self._load_model()
        else:
            print("[SemanticAnalyzer] transformers not installed, using fallback embeddings")

    def _load_model(self):
        """Load CodeBERT model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            print(f"[SemanticAnalyzer] Loaded {self.model_name}")
        except Exception as e:
            print(f"[SemanticAnalyzer] Failed to load model: {e}")
            self.model = None
            self.tokenizer = None

    def get_embedding(self, code: str, cache_key: Optional[str] = None) -> np.ndarray:
        """
        Get semantic embedding for code.

        Args:
            code: Source code
            cache_key: Cache key for this code

        Returns:
            Embedding vector (768-dim for CodeBERT)
        """
        # Check cache
        if cache_key and cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]

        if self.model is None or self.tokenizer is None:
            # Fallback: hash-based embedding
            return self._fallback_embedding(code)

        try:
            # Tokenize
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )

            # Get embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]

            # Cache result
            if cache_key:
                self.embeddings_cache[cache_key] = embedding

            return embedding

        except Exception as e:
            print(f"[SemanticAnalyzer] Embedding failed: {e}")
            return self._fallback_embedding(code)

    def _fallback_embedding(self, code: str) -> np.ndarray:
        """Fallback hash-based embedding when CodeBERT unavailable"""
        # Simple hash-based features
        features = np.zeros(768)

        # Word counts as features
        words = code.split()
        features[0] = len(words)
        features[1] = len(code)
        features[2] = code.count('def ')
        features[3] = code.count('class ')
        features[4] = code.count('import ')
        features[5] = code.count('return ')
        features[6] = code.count('if ')
        features[7] = code.count('for ')
        features[8] = code.count('while ')

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    def compute_similarity(
        self,
        code1: str,
        code2: str,
        cache_key1: Optional[str] = None,
        cache_key2: Optional[str] = None
    ) -> float:
        """
        Compute semantic similarity between two code snippets.

        Args:
            code1: First code snippet
            code2: Second code snippet
            cache_key1: Cache key for code1
            cache_key2: Cache key for code2

        Returns:
            Similarity score (0.0 to 1.0)
        """
        emb1 = self.get_embedding(code1, cache_key1)
        emb2 = self.get_embedding(code2, cache_key2)

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float((similarity + 1.0) / 2.0)  # Map from [-1, 1] to [0, 1]

    def find_similar_modules(
        self,
        target_code: str,
        candidate_codes: Dict[str, str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find modules semantically similar to target.

        Args:
            target_code: Code to find similarities for
            candidate_codes: Dict of {module_name: code}
            top_k: Number of results to return

        Returns:
            List of (module_name, similarity) tuples, sorted by similarity
        """
        target_emb = self.get_embedding(target_code, cache_key=None)

        similarities = []
        for module_name, code in candidate_codes.items():
            emb = self.get_embedding(code, cache_key=module_name)

            # Cosine similarity
            dot_product = np.dot(target_emb, emb)
            norm1 = np.linalg.norm(target_emb)
            norm2 = np.linalg.norm(emb)

            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                similarity = (similarity + 1.0) / 2.0  # Map to [0, 1]
                similarities.append((module_name, float(similarity)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def cluster_modules(
        self,
        module_codes: Dict[str, str],
        num_clusters: int = 5
    ) -> Dict[int, List[str]]:
        """
        Cluster modules by semantic similarity.

        Args:
            module_codes: Dict of {module_name: code}
            num_clusters: Number of clusters

        Returns:
            Dict of {cluster_id: [module_names]}
        """
        if not module_codes:
            return {}

        # Get embeddings for all modules
        embeddings = []
        module_names = []
        for module_name, code in module_codes.items():
            emb = self.get_embedding(code, cache_key=module_name)
            embeddings.append(emb)
            module_names.append(module_name)

        embeddings_matrix = np.array(embeddings)

        # Simple k-means clustering
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(num_clusters, len(module_names)), random_state=42)
            labels = kmeans.fit_predict(embeddings_matrix)

            # Group by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(module_names[i])

            return clusters

        except ImportError:
            print("[SemanticAnalyzer] scikit-learn not available, cannot cluster")
            # Fallback: all in one cluster
            return {0: module_names}


class FunctionalityExtractor:
    """
    Extracts high-level functionality descriptions from code.

    Uses heuristics and patterns to identify what code does.
    """

    def __init__(self):
        self.keyword_patterns = {
            'data_processing': ['parse', 'transform', 'convert', 'format', 'serialize'],
            'database': ['query', 'select', 'insert', 'update', 'delete', 'transaction'],
            'api': ['endpoint', 'route', 'request', 'response', 'http', 'rest'],
            'validation': ['validate', 'check', 'verify', 'ensure', 'assert'],
            'authentication': ['auth', 'login', 'logout', 'token', 'session', 'permission'],
            'testing': ['test', 'mock', 'fixture', 'assert', 'expect'],
            'logging': ['log', 'debug', 'info', 'warn', 'error', 'logger'],
            'caching': ['cache', 'memoize', 'redis', 'memcache'],
            'async': ['async', 'await', 'asyncio', 'concurrent', 'parallel'],
            'ml': ['train', 'predict', 'model', 'embedding', 'neural', 'learning'],
        }

    def extract_functionality(self, code: str, module_name: str = "") -> List[str]:
        """
        Extract functionality tags from code.

        Args:
            code: Source code
            module_name: Module name for additional context

        Returns:
            List of functionality tags
        """
        functionalities = set()

        code_lower = code.lower()
        name_lower = module_name.lower()

        # Check keywords
        for functionality, keywords in self.keyword_patterns.items():
            if any(keyword in code_lower or keyword in name_lower for keyword in keywords):
                functionalities.add(functionality)

        # Check imports for frameworks
        if 'flask' in code_lower or 'fastapi' in code_lower or 'django' in code_lower:
            functionalities.add('api')
            functionalities.add('web')

        if 'pytest' in code_lower or 'unittest' in code_lower:
            functionalities.add('testing')

        if 'sqlalchemy' in code_lower or 'pymongo' in code_lower:
            functionalities.add('database')

        if 'torch' in code_lower or 'tensorflow' in code_lower:
            functionalities.add('ml')

        return sorted(list(functionalities))


# Example usage
if __name__ == "__main__":
    analyzer = SemanticAnalyzer()

    code1 = """
def train_model(data, labels):
    model = NeuralNetwork()
    model.fit(data, labels)
    return model
"""

    code2 = """
def predict(model, input_data):
    predictions = model.predict(input_data)
    return predictions
"""

    code3 = """
def parse_json(json_str):
    data = json.loads(json_str)
    return data
"""

    print("=== Semantic Similarity ===")
    print(f"train vs predict: {analyzer.compute_similarity(code1, code2):.3f}")
    print(f"train vs parse: {analyzer.compute_similarity(code1, code3):.3f}")
    print(f"predict vs parse: {analyzer.compute_similarity(code2, code3):.3f}")

    print("\n=== Functionality Extraction ===")
    extractor = FunctionalityExtractor()
    print(f"code1: {extractor.extract_functionality(code1)}")
    print(f"code2: {extractor.extract_functionality(code2)}")
    print(f"code3: {extractor.extract_functionality(code3)}")
