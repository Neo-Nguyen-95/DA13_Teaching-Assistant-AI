#%% LIB
import faiss
import numpy as np
from business import get_embedding, get_knowledge_text

#%% EMBEDDING

knowledge_text = get_knowledge_text()

knowledge_embeddings = np.array([get_embedding(text) for text in knowledge_text])

dimension = knowledge_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(knowledge_embeddings)

faiss.write_index(index, "knowledge_index.faiss")