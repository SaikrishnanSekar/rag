import nltk
nltk.download('punkt')

def chunk_text(text, chunk_size=100, overlap=20):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        current_chunk.append(sentence)
        current_length += len(sentence.split())

        if current_length >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-overlap:]
            current_length = sum(len(sent.split()) for sent in current_chunk)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks