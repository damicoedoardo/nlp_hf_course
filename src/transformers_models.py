from transformers import pipeline

if __name__ == "__main__":
    # sentiment analysis
    classifier = pipeline("sentiment-analysis")
    out = classifier("I've been waiting for a HuggingFace course my whole life.")
    print(out)
    
    # zero shot classification
    classifier = pipeline("zero-shot-classification")
    out = classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business"],
    )
    print(out)
    
    # text generation
    generator = pipeline("text-generation")
    out = generator("In this course, we will teach you how to")
    print(out)