from openai import OpenAI
import retry
import random


class keyword_generator():
    def __init__(self, api_key, model_name="gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 1024, ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    @retry.retry(tries=5, delay=3)
    def generate_key_word(self, keywords:list[str], gen_num:int=3) -> list[str]:
        response = self.client.chat.completions.create(
            model= self.model_name,
            messages= [
                {"role": "system", "content": f"You are a helpful brainstormer. Given a list of keywords, generate {gen_num} related or similar keywords. Respond with a comma-separated list of keywords."},
                {"role": "user", "content": f"attribute keyword: {keywords}"},
            ],
            temperature= self.temperature,
            max_tokens= self.max_tokens,
        )
        new_keywords_str = response.choices[0].message.content
        new_keywords = new_keywords_str.split(",")
        new_keywords = [keyword.strip() for keyword in new_keywords]
        new_keywords = [''.join(char for char in keyword if char.isalnum() or char.isspace()) for keyword in new_keywords]
        return new_keywords
    
# Example usage
if __name__ == "__main__":

    generator = keyword_generator()

    relation_names = [
        "Locational Relations",
        "Structural Relations",
        "Conceptual Relations",
        "Orientation Relations",
        "Inclusion Relations",
        "Structural Relations"
    ]

    for i in range(20):
        print(f"Attempt {i+1} to generate keywords:")
        # sample some relation names
        choosen_relation_names = random.sample(relation_names, 3)
        new_keywords = generator.generate_key_word(choosen_relation_names)
        for keyword in new_keywords:
            if keyword not in relation_names:
                relation_names.append(keyword)
    print("Final keywords:")
    print(relation_names)

    target_relation_names = [
        "Spatial Relations",
        "Geometric Relations",
        "Functional Relations",
        "Semantic Relations",
        "Directional Relations",
        "Containment Relations",
        "Support Relations"
        ]

    common_relations = set(relation_names) & set(target_relation_names)
    print(f"Number of common elements: {len(common_relations)}")
    print(f"Common elements: {common_relations}")
