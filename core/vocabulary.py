"""
Vocabulary, CTC Decoding & Language Model Rescoring Module for LipNet VSR.
Includes:
1. CTC Greedy Best-Path Decoder.
2. CTC Prefix Beam Search.
3. Language Model (LM) & Lexicon Grammar Beam Rescorer for resolving viseme ambiguities.
"""

from typing import List, Dict, Tuple, Set, Optional
import re
import numpy as np


# Standard GRID & Conversational English Lexicon
GRID_COMMANDS = {"bin", "lay", "place", "set"}
GRID_COLORS = {"blue", "green", "red", "white"}
GRID_PREPOSITIONS = {"at", "by", "in", "with"}
GRID_LETTERS = set("abcdefghijklmnopqrstuvwxyz")
GRID_DIGITS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", "again", "soon", "now", "please"}
GRID_ADVERBS = {"again", "now", "please", "soon"}

COMMON_ENGLISH_WORDS = GRID_COMMANDS | GRID_COLORS | GRID_PREPOSITIONS | GRID_DIGITS | GRID_ADVERBS | {
    # Greetings & Farewells
    "hello", "hi", "hey", "bye", "goodbye", "welcome",
    # Common Responses
    "yes", "no", "ok", "okay", "sure", "right", "thanks", "sorry",
    # Conversational
    "thank", "you", "good", "morning", "afternoon", "evening", "night",
    "how", "are", "what", "is", "your", "name", "nice", "to", "meet",
    "please", "help", "stop", "start", "play", "go", "come", "wait",
    "can", "do", "will", "have", "has", "had", "was", "were", "been",
    "the", "and", "for", "not", "but", "this", "that", "with", "from",
    "my", "me", "we", "our", "they", "them", "he", "she", "it",
    # Project-specific
    "world", "speech", "visual", "recognition", "learning", "deep",
    "camera", "video", "face", "mouth", "audio", "lip", "read", "reading",
    # Numbers as words
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    # Common short words that may appear
    "a", "i", "am", "an", "as", "at", "be", "by", "if", "in", "of", "on", "or", "so", "up",
    # Additional conversational
    "say", "see", "look", "know", "think", "want", "need", "like", "love",
    "time", "day", "work", "call", "try", "give", "tell", "make", "take",
    "here", "there", "where", "when", "why", "who", "which"
}


class Vocabulary:
    """
    Manages character vocabulary, token mappings, CTC beam decoding,
    and Language Model beam rescoring.
    """
    def __init__(self, vocab_chars: str = "abcdefghijklmnopqrstuvwxyz' 0123456789"):
        self.vocab_chars = list(vocab_chars)
        self.blank_token = "<blank>"
        self.tokens = [self.blank_token] + self.vocab_chars
        
        self.char_to_id: Dict[str, int] = {char: idx for idx, char in enumerate(self.tokens)}
        self.id_to_char: Dict[int, str] = {idx: char for idx, char in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)
        self.blank_idx = 0
        self.lexicon: Set[str] = COMMON_ENGLISH_WORDS

    def encode(self, text: str) -> List[int]:
        """Convert string to list of integer token IDs."""
        text = text.lower().strip()
        encoded = []
        for ch in text:
            if ch in self.char_to_id:
                encoded.append(self.char_to_id[ch])
            elif ch == '\n' or ch == '\t':
                encoded.append(self.char_to_id.get(' ', 0))
        return encoded

    def ctc_greedy_decode(self, logits_or_probs: np.ndarray) -> str:
        """
        Performs CTC Greedy Best-Path decoding.
        """
        best_path = np.argmax(logits_or_probs, axis=-1)
        decoded_tokens = []
        prev_token = None
        
        for token in best_path:
            if token != prev_token:
                if token != self.blank_idx:
                    decoded_tokens.append(token)
            prev_token = token
            
        decoded_str = "".join([self.id_to_char.get(idx, "") for idx in decoded_tokens])
        return decoded_str.strip()

    def ctc_beam_search_decode(
        self,
        probs: np.ndarray,
        beam_width: int = 15,
        lm_weight: float = 0.65
    ) -> str:
        """
        Performs CTC Beam Search with prefix tree management and Language Model rescoring.
        """
        T, V = probs.shape
        # beams: {prefix_tuple: (p_blank, p_non_blank)}
        beams = {(): (1.0, 0.0)}

        for t in range(T):
            new_beams = {}
            for prefix, (p_b, p_nb) in beams.items():
                p_total = p_b + p_nb
                
                # 1. Blank token transition
                p_blank_prob = probs[t, self.blank_idx]
                if prefix not in new_beams:
                    new_beams[prefix] = (0.0, 0.0)
                n_pb, n_pnb = new_beams[prefix]
                new_beams[prefix] = (n_pb + p_total * p_blank_prob, n_pnb)
                
                # 2. Non-blank token transitions
                for v in range(1, V):
                    p_char = probs[t, v]
                    if p_char < 1e-5:
                        continue
                    
                    new_prefix = prefix + (v,)
                    if prefix and prefix[-1] == v:
                        if new_prefix not in new_beams:
                            new_beams[new_prefix] = (0.0, 0.0)
                        n_pb, n_pnb = new_beams[new_prefix]
                        new_beams[new_prefix] = (n_pb, n_pnb + p_b * p_char)
                        
                        n_pb_same, n_pnb_same = new_beams[prefix]
                        new_beams[prefix] = (n_pb_same, n_pnb_same + p_nb * p_char)
                    else:
                        if new_prefix not in new_beams:
                            new_beams[new_prefix] = (0.0, 0.0)
                        n_pb, n_pnb = new_beams[new_prefix]
                        new_beams[new_prefix] = (n_pb, n_pnb + p_total * p_char)

            # Sort and prune beams
            sorted_beams = sorted(
                new_beams.items(),
                key=lambda x: x[1][0] + x[1][1],
                reverse=True
            )
            beams = dict(sorted_beams[:beam_width])

        # Score top candidate hypotheses using Language Model & Lexicon
        candidate_hyps = []
        for prefix, (pb, pnb) in beams.items():
            raw_text = "".join([self.id_to_char.get(idx, "") for idx in prefix]).strip()
            if not raw_text:
                continue
            ctc_score = np.log(max(1e-12, pb + pnb))
            lm_score = self._compute_lm_score(raw_text)
            total_score = ctc_score + lm_weight * lm_score
            candidate_hyps.append((total_score, raw_text))

        if candidate_hyps:
            candidate_hyps.sort(key=lambda x: x[0], reverse=True)
            best_raw = candidate_hyps[0][1]
            return self._clean_and_postprocess_transcript(best_raw)

        return ""

    def _compute_lm_score(self, text: str) -> float:
        """
        Computes Lexicon and N-gram score for a candidate text hypothesis.
        Rewards recognized valid dictionary words and sentence structure.
        """
        words = text.lower().split()
        if not words:
            return -10.0

        score = 0.0
        for w in words:
            if w in self.lexicon:
                score += 2.0  # Reward dictionary word
            elif len(w) == 1 and w.isalpha():
                score += 1.0  # Single letter spelled out (GRID format)
            else:
                # Penalty for non-dictionary fragments
                score -= 1.5

        # Reward word count density
        score += 0.5 * len(words)
        return score

    def _clean_and_postprocess_transcript(self, text: str) -> str:
        """
        Post-processes raw decoded transcript:
        1. Collapses repetitive character patterns (e.g. 'vgvgvg' -> 'vg', 'aaaa' -> 'a').
        2. Filters out random single-character noise.
        3. Snaps close viseme phonemes to valid English words in the lexicon.
        """
        # Collapse repeated character runs >= 2
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        # Collapse repeated 2-char n-grams (e.g. 'vgvgvg' -> 'vg')
        text = re.sub(r'(.{2})\1{2,}', r'\1', text)
        
        # Collapse multiple spaces
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()
        words = cleaned.split()
        
        if not words:
            return ""

        corrected_words = []
        for w in words:
            w_lower = w.lower()
            if w_lower in self.lexicon:
                corrected_words.append(w_lower)
            elif len(w_lower) == 1 and w_lower.isalpha():
                # Single letters are valid (GRID letter spelling)
                corrected_words.append(w_lower)
            else:
                best_match = self._find_closest_lexicon_word(w_lower)
                if best_match:
                    corrected_words.append(best_match)
                elif len(w_lower) >= 2:
                    # Keep non-dictionary words rather than discarding useful output
                    corrected_words.append(w_lower)

        result = " ".join(corrected_words).strip()
        return result if result else ""

    def _find_closest_lexicon_word(self, word: str) -> Optional[str]:
        """Finds closest dictionary word within edit distance threshold.
        Uses distance 1 for short words (<=4 chars) and distance 2 for longer words."""
        if len(word) < 2:
            return None
        
        max_dist = 1 if len(word) <= 4 else 2
        best_candidate = None
        best_dist = max_dist + 1
        
        for target in self.lexicon:
            if abs(len(target) - len(word)) > max_dist:
                continue
            dist = self._levenshtein(word, target)
            if dist < best_dist:
                best_dist = dist
                best_candidate = target

        return best_candidate if best_dist <= max_dist else None

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return Vocabulary._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]


# Default global vocabulary instance for LipNet
default_vocab = Vocabulary()
