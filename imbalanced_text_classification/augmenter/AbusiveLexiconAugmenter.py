import fasttext
import json
import math
import numpy as np
import torch

class AbusiveLexiconAugmenter(): 
    def __init__(self, aug_p, top_k):
        lexicon_path = 'data/augmentation_src/abusive_language_lexicon/abusive_lexicon.json'
        with open(lexicon_path) as f:
            self.lexicon = json.load(f)
        fasttext_vectors_path = "data/augmentation_src/abusive_language_lexicon/abusive_lexicon_fasttext_vectors.npy"
        with open(fasttext_vectors_path, 'rb') as f:
            self.lexicon_vectors = np.load(f)
        fasttext_model_path = '/mounts/Users/cisintern/zhangyaq/imbalanced_text_classification/data/augmentation_src/abusive_language_lexicon/cc.en.300.bin'
        self.fasttext_model = fasttext.load_model(fasttext_model_path)
        self.aug_p = aug_p
        self.top_k = top_k

    def check_if_text_contain_lexicon_words(self, text):
        text = text.lower()
        contained_words = [word for word in self.lexicon if word in text]
        return len(contained_words) > 0
    
    def augment(self, texts, n=1):
        augmented_texts = []
        for text in texts:
            # print(f"\nAugmenting: text = '{text}'")
            text = text.lower()
            contained_words = []
            # first split text into tokens with spaces, check intersection and replace them, in case a whole subword in text is matched to a term in lexicon
            text_placeholdered = []
            for word in text.split():
                if word in self.lexicon:
                    text_placeholdered.append(self._get_placeholder_for_index(lexicon_i=self.lexicon.index(word), contained_words_i=len(contained_words)))
                    contained_words.append(word)
                else:
                    text_placeholdered.append(word)
            text_placeholdered = " ".join(text_placeholdered)
            # Then check strings match (including empty space)
            for i, word in enumerate(self.lexicon):
                if word in text_placeholdered:
                    text_placeholdered = text_placeholdered.replace(word, self._get_placeholder_for_index(lexicon_i=self.lexicon.index(word), contained_words_i=len(contained_words)))
                    contained_words.append(word)
            # print(f"- Contained words in abusive lexicon: {contained_words}")
            # print(f"- Text with placeholder: text = '{text_placeholdered}'")

            num_aug = math.ceil(len(contained_words) * self.aug_p)
            # print(f"Numbers of tokens for augmentation = {num_aug}")
            augmented_words_index = torch.randint(0, len(contained_words), (num_aug,))
            # print(f"- index of word to augment in contained words list = {augmented_words_index}")
            augmented_words = [contained_words[index] for index in augmented_words_index]
            # print(f"- words to be augmented: {augmented_words}")
            # Recover the placeholder of the word that is not choosen to be augmented
            for contained_words_i, word in enumerate(contained_words):
                if contained_words_i not in augmented_words_index:
                    text_placeholdered = text_placeholdered.replace(self._get_placeholder_for_index(lexicon_i=self.lexicon.index(word), 
                                                                                                    contained_words_i=contained_words_i), 
                                                                    word)
            # print(f"- Only words to be augmented need placeholder: text = '{text_placeholdered}'")

            text_augmented = text_placeholdered
            for aug_word_index in augmented_words_index:
                aug_word = contained_words[aug_word_index]
                # print(f"- Augmenting word '{aug_word}'")
                fasttext_vec = self._get_fasttext_vector(aug_word)
                # print(f"-- FastText vector = {fasttext_vec.shape}")
                top_k_indices_in_lexicon = self._top_k_most_similar_lexicon_vec_index(fasttext_vec)
                # print(f"-- The indicies of {self.top_k} most similar words in the lexicon are: {top_k_indices_in_lexicon} = {[self.lexicon[i] for i in top_k_indices_in_lexicon]}")
                aug_word_lexicon_index = self.lexicon.index(aug_word)
                # print(f"-- Needs to remove lexicon[{aug_word_lexicon_index}]={aug_word}({self.lexicon[aug_word_lexicon_index]})")
                top_k_indices_in_lexicon = [i for i in top_k_indices_in_lexicon if i != aug_word_lexicon_index]
                # print(f"-- Now choose from {top_k_indices_in_lexicon}")
                replace_lexicon_word_index = torch.randint(0, self.top_k, (1,)).tolist()[0]
                # print(f"-- Choose the word with index {replace_lexicon_word_index} in {top_k_indices_in_lexicon}")
                replace_lexicon_word = self.lexicon[top_k_indices_in_lexicon[replace_lexicon_word_index]]
                # print(f"-- The word to replace '{aug_word}' is '{replace_lexicon_word}'")
                text_augmented = text_augmented.replace(self._get_placeholder_for_index(lexicon_i=self.lexicon.index(aug_word),
                                                                                            contained_words_i=aug_word_index), 
                                                                                            replace_lexicon_word)
                # print(f"text = '{text_augmented}'")
            augmented_texts.append(text_augmented)
        return augmented_texts

    def _top_k_most_similar_lexicon_vec_index(self, vec):
        assert self.lexicon_vectors.shape == (3331, 300)
        dot_product = np.dot(vec, self.lexicon_vectors.T)
        norm_a = np.linalg.norm(vec)
        norm_b = np.linalg.norm(self.lexicon_vectors, axis=1)
        score = dot_product / (norm_a * norm_b)
        top_k_indices = score.argsort()[-self.top_k-1:][::-1] # the top most similar words also include the original word 
        return top_k_indices
    
    def _get_fasttext_vector(self, term):
        return np.mean([self.fasttext_model.get_word_vector(word) for word in term.split()], axis=0)
    
    def _get_placeholder_for_index(self, lexicon_i, contained_words_i):
        return f"[REPLACE_LEXICON[{lexicon_i}]_CONTAINEDWORDS[{contained_words_i}]]"
    
# aug = AbusiveLexiconAugmenter(aug_p=0.6, top_k=3)
# # aug.check_if_text_contain_lexicon_words("Sorry, just want to check if abusive is contained in the abusive lexicon") # True

# texts = ["if bitch can be shocker found shocker"]
# aug.augment(texts)