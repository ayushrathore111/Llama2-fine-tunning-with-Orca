from datasets import load_dataset, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from huggingface_hub import login
data= 'Open-Orca/OpenOrca'

dataset = load_dataset(data)

#4231945 rows use only 1000 rows to tunning...
dataset = dataset.select(range(1000))
print(dataset)

#apply filter...

filtered_examples= [x for x in dataset if len(x['response'].split()) >= 100]

# Combine system_prompt and question..
combined_texts = [x['system_prompt'] + " " + x['question'] for x in filtered_examples]

# Vectorize....
vectorizer = TfidfVectorizer().fit(combined_texts)
vectors = vectorizer.transform(combined_texts)

# cosine similarity matrix...
cosine_sim = cosine_similarity(vectors)

# Identify duplicates....
duplicates = set()
for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i][j] > 0.95:
            duplicates.add(j)

# Filter out duplicates...
deduplicated_examples = [x for idx, x in enumerate(filtered_examples) if idx not in duplicates]

# Create deduplicated dataset....
deduplicated_hf_dataset = Dataset.from_dict({
    "id": [x['id'] for x in deduplicated_examples],
    "system_prompt": [x['system_prompt'] for x in deduplicated_examples],
    "question": [x['question'] for x in deduplicated_examples],
    "response": [x['response'] for x in deduplicated_examples]
})

print(deduplicated_hf_dataset)
#login to huggingface...
login(token="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")

deduplicated_hf_dataset.push_to_hub("ar111/modified_orca_dataset")

