import PyPDF2
import re
import spacy
from spacy.training.example import Example

# Function to extract text from PDF
def fetch_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Function to preprocess text
def preprocess_text(text):
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    return text

# Function to generate weak labels
def generate_weak_labels(text):
    entities = []
    date_pattern = re.compile(r'\b(?:\d{1,2}[-/th|st|nd|rd\s.]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\s.-]*\d{1,2}[-/th|st|nd|rd\s,.]*\d{2,4}\b', re.IGNORECASE)
    dates = date_pattern.finditer(text)
    for match in dates:
        entities.append((match.start(), match.end(), "EFFECTIVE_DATE"))

    vendor_pattern = re.compile(r'\b(?:Inc|Corp|Ltd|LLC)\b', re.IGNORECASE)
    vendors = vendor_pattern.finditer(text)
    for match in vendors:
        entities.append((match.start(), match.end(), "VENDOR_NAME"))

    return {"entities": entities}

# Load and preprocess PDFs
corpus = []
books = [
  {"title": "MSA1", "pdf_link": "MSA_1.pdf"},
    {"title": "MSA2", "pdf_link": "MSA_2.pdf"},
    {"title": "MSA3", "pdf_link": "MSA_3.pdf"},
    {"title": "MSA4", "pdf_link": "MSA_4.pdf"},
    {"title": "MSAA1", "pdf_link": "MSAA_1.pdf"},
    {"title": "MSAA2", "pdf_link": "MSAA_2.pdf"},
    {"title": "NDA1", "pdf_link": "NDA_1.pdf"},
    {"title": "NDA2", "pdf_link": "NDA_2.pdf"},
    {"title": "NDA3", "pdf_link": "NDA_3.pdf"},
    {"title": "NDA4", "pdf_link": "NDA_4.pdf"},
    {"title": "SOW1", "pdf_link": "SOW_1.pdf"},
    {"title": "SOW2", "pdf_link": "SOW_2.pdf"},
    {"title": "SOW3", "pdf_link": "SOW_3.pdf"},
    {"title": "SOW4", "pdf_link": "SOW_4.pdf"},

    # Add more PDFs here
]

for book in books:
    pdf_link = book["pdf_link"]
    text = fetch_text_from_pdf(pdf_link)
    if text:
        preprocessed_text = preprocess_text(text)
        corpus.append(preprocessed_text)

# Generate weak labels
TRAIN_DATA = [(text, generate_weak_labels(text)) for text in corpus]

# Load pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

# Add labels to the NER component
ner = nlp.get_pipe("ner")
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Initialize the optimizer
optimizer = nlp.resume_training()

# Train the model
for i in range(10):
    losses = {}
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], losses=losses, drop=0.35, sgd=optimizer)
    print(f"Losses at iteration {i}: {losses}")

# Save the trained model
nlp.to_disk("ner_model")

# Load the fine-tuned model
nlp = spacy.load("ner_model")

# Prompt user for input and provide answers
while True:
    user_input = input("Enter a query related to contract documents (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    doc = nlp(user_input)
    print("Entities detected:")
    for ent in doc.ents:
        print(f"{ent.text} -> {ent.label_}")
