"""Semantic chunking using sentence embeddings."""

import warnings
from typing import List, Union, Literal

import numpy as np

from chonkie.chunker.base import BaseChunker
from chonkie.embeddings.base import BaseEmbeddings
from chonkie.types import SemanticChunk, SemanticSentence, Sentence


class SemanticChunker(BaseChunker):
    """Chunker that splits text into semantically coherent chunks using embeddings.

    Args:
        embedding_model: Embedding model to use for semantic chunking
        mode: Mode for grouping sentences, either "cumulative" or "window"
        threshold: Threshold for semantic similarity (0-1) or percentile (1-100), defaults to "auto"
        chunk_size: Maximum tokens allowed per chunk
        similarity_window: Number of sentences to consider for similarity threshold calculation
        min_sentences: Minimum number of sentences per chunk
        min_characters_per_sentence: Minimum number of characters per sentence
        min_chunk_size: Minimum number of tokens per sentence (defaults to 2)
        threshold_step: Step size for similarity threshold calculation
        delim: Delimiters to split sentences on
        return_type: Whether to return chunks or texts
        verbose: Whether to print debug information during chunking
    
    Raises:
        ValueError: If parameters are invalid
    """

    def __init__(
        self,
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-base-8M",
        mode: str = "window",
        threshold: Union[str, float, int] = "auto",
        chunk_size: int = 512,
        similarity_window: int = 1,
        min_sentences: int = 1,
        min_chunk_size: int = 2,
        min_characters_per_sentence: int = 12,
        threshold_step: float = 0.01,
        delim: Union[str, List[str]] = [".", "!", "?", "\n"],
        return_type: Literal["chunks", "texts"] = "chunks",
        verbose: bool = False,
        **kwargs
    ):
        """Initialize the SemanticChunker.

        SemanticChunkers split text into semantically coherent chunks using embeddings.

        Args:
            embedding_model: Name of the sentence-transformers model to load
            mode: Mode for grouping sentences, either "cumulative" or "window"
            threshold: Threshold for semantic similarity (0-1) or percentile (1-100), defaults to "auto"
            chunk_size: Maximum tokens allowed per chunk
            similarity_window: Number of sentences to consider for similarity threshold calculation
            min_sentences: Minimum number of sentences per chunk
            min_characters_per_sentence: Minimum number of characters per sentence
            min_chunk_size: Minimum number of tokens per chunk (and sentence, defaults to 2)
            threshold_step: Step size for similarity threshold calculation
            delim: Delimiters to split sentences on
            return_type: Whether to return chunks or texts
            verbose: Whether to print debug information during chunking
            **kwargs: Additional keyword arguments

        Raises:
            ValueError: If parameters are invalid
            ImportError: If required dependencies aren't installed

        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if min_sentences <= 0:
            raise ValueError("min_sentences must be positive")
        if similarity_window < 0:
            raise ValueError("similarity_window must be non-negative")
        if threshold_step <= 0 or threshold_step >= 1:
            raise ValueError("threshold_step must be between 0 and 1")
        if mode not in ["cumulative", "window"]:
            raise ValueError("mode must be 'cumulative' or 'window'")
        if type(threshold) not in [str, float, int]:
            raise ValueError("threshold must be a string, float, or int")
        if type(delim) not in [str, list]:
            raise ValueError("delim must be a string or list of strings")
        elif type(threshold) == str and threshold not in ["auto"]:
            raise ValueError("threshold must be 'auto', 'smart', or 'percentile'")
        elif type(threshold) == float and (threshold < 0 or threshold > 1):
            raise ValueError("threshold (float) must be between 0 and 1")
        elif type(threshold) == int and (threshold < 1 or threshold > 100):
            raise ValueError("threshold (int) must be between 1 and 100")
        if return_type not in ["chunks", "texts"]:
            raise ValueError("Invalid return_type. Must be either 'chunks' or 'texts'.")

        self.mode = mode
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.similarity_window = similarity_window if mode == "window" else 1
        self.min_sentences = min_sentences
        self.min_chunk_size = min_chunk_size
        self.min_characters_per_sentence = min_characters_per_sentence
        self.threshold_step = threshold_step
        self.delim = delim
        self.sep = "🦛"
        self.return_type = return_type
        self.verbose = verbose
        
        if isinstance(threshold, float):
            self.similarity_threshold = threshold
            self.similarity_percentile = None
        elif isinstance(threshold, int):
            self.similarity_percentile = threshold
            self.similarity_threshold = None
        else:
            self.similarity_threshold = None
            self.similarity_percentile = None

        if isinstance(embedding_model, BaseEmbeddings):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            from chonkie.embeddings.auto import AutoEmbeddings

            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model, **kwargs)
        else:
            raise ValueError(
                f"{embedding_model} is not a valid embedding model"
            )

        # Probably the dependency is not installed
        if self.embedding_model is None:
            raise ImportError(
                f"{embedding_model} is not a valid embedding model",
                "Please install the `semantic` extra to use this feature",
            )

        # Keeping the tokenizer the same as the sentence model is important
        # for the group semantic meaning to be calculated properly
        super().__init__(self.embedding_model.get_tokenizer_or_token_counter())

        # Remove the multiprocessing flag from the base class
        self._use_multiprocessing = False

    def _split_sentences(
        self,
        text: str,
    ) -> List[str]:
        """Fast sentence splitting while maintaining accuracy.

        This method is faster than using regex for sentence splitting and is more accurate 
        than using the spaCy sentence tokenizer.

        Args:
            text: Input text to be split into sentences

        Returns:
            List of sentences
        """
        if not text:
            return []

        # Add separator after each delimiter sequence
        t = text
        i = 0
        while i < len(t):
            # Find end of delimiter sequence
            j = i
            while j < len(t) and any(t[j] == d for d in self.delim):
                j += 1
            
            # If we found a delimiter sequence
            if j > i:
                # Add separator after the sequence
                t = t[:j] + self.sep + t[j:]
                i = j + len(self.sep)
            else:
                i += 1

        # Initial split
        splits = [s for s in t.split(self.sep) if s != ""]

        # Combine short splits with previous sentence
        sentences = []
        current = ""

        for s in splits:
            if len(s.strip()) < self.min_characters_per_sentence:
                current += s
            else:
                if current:
                    sentences.append(current)
                current = s

        if current:
            sentences.append(current)

        return sentences

    def _compute_similarity_threshold(self, all_similarities: List[float]) -> float:
        """Compute similarity threshold based on percentile if specified."""
        if self.similarity_threshold is not None:
            return self.similarity_threshold
        else:
            return float(np.percentile(all_similarities, self.similarity_percentile))

    def _prepare_sentences(self, text: str) -> List[Sentence]:
        """Prepare sentences with precomputed information.

        Args:
            text: Input text to be processed

        Returns:
            List of Sentence objects with precomputed token counts and embeddings

        """
        if not text.strip():
            return []

        # Split text into sentences
        raw_sentences = self._split_sentences(text)

        # Get start and end indices for each sentence
        sentence_indices = []
        current_idx = 0
        for sent in raw_sentences:
            start_idx = text.find(sent, current_idx)
            end_idx = start_idx + len(sent)
            sentence_indices.append((start_idx, end_idx))
            current_idx = end_idx

        # Batch compute embeddings for all sentences
        sentence_groups = []
        window_size = self.similarity_window
        
        for i in range(len(raw_sentences)):
            # Calculate window bounds
            start = max(0, i - window_size)
            end = min(len(raw_sentences), i + window_size + 1)
            
            # Create group from window
            group = raw_sentences[start:end]
            sentence_groups.append("".join(group))
        embeddings = self.embedding_model.embed_batch(sentence_groups)

        # Batch compute token counts
        token_counts = self._count_tokens_batch(raw_sentences)

        # Create Sentence objects with all precomputed information
        sentences = [
            SemanticSentence(
                text=sent,
                start_index=start_idx,
                end_index=end_idx,
                token_count=count,
                embedding=embedding,
            )
            for sent, (start_idx, end_idx), count, embedding in zip(
                raw_sentences, sentence_indices, token_counts, embeddings
            )
        ]

        return sentences

    def _get_semantic_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        similarity = self.embedding_model.similarity(embedding1, embedding2)
        return similarity

    def _compute_group_embedding(self, sentences: List[Sentence]) -> np.ndarray:
        """Compute mean embedding for a group of sentences."""
        if len(sentences) == 1:
            return sentences[0].embedding
        else:
            #NOTE: There's a known issue, where while calculating the sentence embeddings special tokens are added
            # but when taking the token count the special tokens are not being counted, which causes a mismatch here. 
            # This is a known issue and we're working on a fix. At the moment, the error is minimal and doesn't affect the chunking as much. 

            #TODO: Account for embedding model truncating to max_seq_length, which causes a mismatch in the token count.
            return np.divide(
                np.sum([(sent.embedding * sent.token_count) for sent in sentences], axis=0),
                np.sum([sent.token_count for sent in sentences]),
                dtype=np.float32,
            )

    def _compute_window_similarities(self, sentences: List[Sentence]) -> List[float]:
        """Compute similarities between sentences using a sliding window approach.
        
        For each sentence i, computes similarity between its embedding and the
        combined embedding of the previous similarity_window sentences.
        """
        if not sentences:
            return []
        
        similarities = [1.0]  # First sentence always has similarity 1.0
        
        # Initialize window with first sentence
        window = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Update window - remove oldest if window is full
            if len(window) >= self.similarity_window:
                window.pop(0)
            window.append(sentences[i-1])
            
            # Compute window embedding
            window_embedding = self._compute_group_embedding(window)
            
            # Compare current sentence with window
            similarity = self._get_semantic_similarity(
                window_embedding,
                sentences[i].embedding
            )
            similarities.append(similarity)
        
        return similarities

    def _get_split_indices(
        self, similarities: List[float], threshold: float = None
    ) -> List[int]:
        """Get indices of sentences to split at."""
        if threshold is None:
            threshold = (
                self.similarity_threshold
                if self.similarity_threshold is not None
                else 0.5
            )

        # Include start index
        splits = [0]
        
        # Add split points where similarity drops below threshold
        for i, similarity in enumerate(similarities[1:], 1):
            if similarity <= threshold:
                splits.append(i)
            
        # Add end index if not already included
        if splits[-1] != len(similarities):
            splits.append(len(similarities))
        
        # Filter out splits that would create chunks smaller than min_sentences
        filtered_splits = [splits[0]]  # Always keep start
        for i in range(1, len(splits)):
            if splits[i] - filtered_splits[-1] >= self.min_sentences:
                filtered_splits.append(splits[i])
            
        return filtered_splits

    def _validate_sentence_sizes(self, sentences: List[Sentence]) -> None:
        """Validate that no individual sentence exceeds the maximum chunk size."""
        if self.verbose:
            print("\nValidating sentence sizes:")
            for i, sent in enumerate(sentences):
                print(f"Sentence {i}: {sent.token_count} tokens - {sent.text!r}")
        
        oversized = [
            (i, sent) for i, sent in enumerate(sentences)
            if sent.token_count > self.chunk_size
        ]
        if oversized:
            if self.verbose:
                print(f"\nFound {len(oversized)} oversized sentences!")
            
            examples = [
                f"Sentence {i} ({sent.token_count} tokens): {sent.text[:50]}..."
                for i, sent in oversized[:3]
            ]
            msg = (
                f"Found {len(oversized)} sentences exceeding maximum chunk size ({self.chunk_size}).\n"
                f"Example oversized sentences:\n" + "\n".join(examples)
            )
            if self.verbose:
                print(f"Error message:\n{msg}")
            raise ValueError(msg)

    def _calculate_threshold_via_binary_search(
        self, 
        sentences: List[Sentence], 
        max_iterations: int = 10,
        verbose: bool = None
    ) -> float:
        """Calculate similarity threshold via binary search.
        
        Args:
            sentences: List of Sentence objects containing token counts and embeddings
            max_iterations: Maximum number of binary search iterations (default: 10)
            verbose: Override class-level verbose setting (default: None)
            
        Returns:
            float: Optimal similarity threshold for chunking
            
        Raises:
            ValueError: If sentences list is empty or if all sentences are too large
            ValueError: If max_iterations is less than 1
        """
        # At the start of the method, resolve verbose setting
        verbose = self.verbose if verbose is None else verbose
        
        if not sentences:
            raise ValueError("Cannot calculate threshold for empty sentence list")
        
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        
        # Remove oversized sentence validation from here since it's now handled in chunk()
        
        # Compute initial statistics
        similarities = self._compute_window_similarities(sentences)
        median = np.median(similarities)
        std = np.std(similarities)
        
        # Set search boundaries within 1 standard deviation
        low = max(median - std, 0.0)
        high = min(median + std, 1.0)
        best_threshold = (low + high) / 2
        best_score = float('inf')
        
        # Get token information once, outside the loop
        token_counts = [sent.token_count for sent in sentences]
        cumulative_tokens = np.cumsum([0] + token_counts)
        
        def check_chunk_sizes(split_counts: np.ndarray) -> tuple[float, bool, str]:
            """Evaluate chunk sizes and return a score.
            
            Returns:
                tuple[float, bool, str]: (score, is_valid, reason)
                - score: Lower is better, 0 is perfect
                - is_valid: Whether this is a valid solution
                - reason: Description of why the score was assigned
            """
            if len(split_counts) == 0:
                return float('inf'), False, "No chunks created"
            
            # Count violations
            too_large = sum(1 for count in split_counts if count > self.chunk_size)
            too_small = sum(1 for count in split_counts if count < self.min_chunk_size)
            
            # Calculate average deviation from ideal size
            ideal_size = (self.min_chunk_size + self.chunk_size) / 2
            size_deviations = [abs(count - ideal_size) for count in split_counts]
            avg_deviation = np.mean(size_deviations) if size_deviations else float('inf')
            
            # Calculate score (weighted sum of violations and deviations)
            score = (too_large * 1000 + too_small * 100 + avg_deviation)
            
            is_valid = too_large == 0 and too_small == 0
            reason = f"{len(split_counts)} chunks: {too_large} too large, {too_small} too small, avg deviation {avg_deviation:.1f}"
            
            return score, is_valid, reason
        
        # Binary search loop
        for iteration in range(max_iterations):
            threshold = (low + high) / 2
            
            # Get splits and calculate chunk sizes
            split_indices = self._get_split_indices(similarities, threshold)
            split_token_counts = np.diff(cumulative_tokens[split_indices])
            
            # Evaluate this solution
            score, is_valid, reason = check_chunk_sizes(split_token_counts)
            
            # Track best solution
            if score < best_score:
                best_score = score
                best_threshold = threshold
                best_reason = reason
            
            # Log progress for debugging
            if verbose:
                print(f"Iteration {iteration + 1}: threshold={threshold:.3f}, {reason}")
            
            # Perfect solution found
            if is_valid:
                return threshold
            
            # Calculate average chunk size to guide the search
            avg_chunk_size = np.mean(split_token_counts)
            
            # Adjust bounds based on average chunk size
            if avg_chunk_size > self.chunk_size:
                # Chunks too large, increase threshold to split more
                low = threshold + self.threshold_step
            elif avg_chunk_size < self.min_chunk_size:
                # Chunks too small, decrease threshold to combine more
                high = threshold - self.threshold_step
            else:
                # Size is okay, but not valid - try both directions
                if score > best_score:
                    # Last change made things worse, reverse direction
                    if threshold > best_threshold:
                        high = threshold - self.threshold_step
                    else:
                        low = threshold + self.threshold_step
            
            # Check if search range is too small
            if abs(high - low) <= self.threshold_step:
                break
        
        # If we didn't find a perfect solution, warn and return best attempt
        warnings.warn(
            f"Could not find perfect threshold after {max_iterations} iterations. "
            f"Using best approximation: {best_reason}",
            stacklevel=2,
        )
        
        return best_threshold

    def _calculate_threshold_via_percentile(self, sentences: List[Sentence]) -> float:
        """Calculate similarity threshold via percentile."""
        # Compute all pairwise similarities, since the embeddings are already computed
        # The embeddings are computed assuming a similarity window is applied
        all_similarities = self._compute_window_similarities(sentences)
        return float(np.percentile(all_similarities, 100 - self.similarity_percentile))

    def _calculate_similarity_threshold(self, sentences: List[Sentence]) -> float:
        """Calculate similarity threshold either through the smart binary search or percentile."""
        if self.similarity_threshold is not None:
            return self.similarity_threshold
        elif self.similarity_percentile is not None:
            return self._calculate_threshold_via_percentile(sentences)
        else:
            return self._calculate_threshold_via_binary_search(sentences)

    def _group_sentences_cumulative(
        self, sentences: List[Sentence]
    ) -> List[List[Sentence]]:
        """Group sentences based on semantic similarity, ignoring token count limits.

        Args:
            sentences: List of Sentence objects with precomputed embeddings

        Returns:
            List of sentence groups, where each group is semantically coherent

        """
        groups = []
        current_group = sentences[: self.min_sentences]
        current_embedding = self._compute_group_embedding(current_group)

        for sentence in sentences[self.min_sentences :]:
            # Compare new sentence against mean embedding of entire current group
            similarity = self._get_semantic_similarity(
                current_embedding, sentence.embedding
            )

            if similarity >= self.similarity_threshold:
                # Add to current group
                current_group.append(sentence)
                # Update mean embedding
                current_embedding = self._compute_group_embedding(current_group)
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [sentence]
                current_embedding = sentence.embedding

        # Add final group
        if current_group:
            groups.append(current_group)

        return groups

    def _group_sentences_window(
        self, sentences: List[Sentence]
    ) -> List[List[Sentence]]:
        """Group sentences based on semantic similarity, respecting the similarity window."""
        similarities = self._compute_window_similarities(sentences) # NOTE: This is calculating pairwise, but not window. 
        split_indices = self._get_split_indices(similarities, self.similarity_threshold)
        groups = [
            sentences[split_indices[i] : split_indices[i + 1]]
            for i in range(len(split_indices) - 1)
        ]
        return groups

    def _group_sentences(self, sentences: List[Sentence]) -> List[List[Sentence]]:
        """Group sentences based on semantic similarity, either cumulatively or by window."""
        if self.mode == "cumulative":
            return self._group_sentences_cumulative(sentences)
        else:
            return self._group_sentences_window(sentences)

    def _create_chunk(
        self, sentences: List[Sentence]
    ) -> SemanticChunk:
        """Create a chunk from a list of sentences."""
        if not sentences:
            raise ValueError("Cannot create chunk from empty sentence list")
        if self.return_type == "chunks":
            # Compute chunk text and token count from sentences
            text = "".join(sent.text for sent in sentences)
            token_count = sum(sent.token_count for sent in sentences)
            return SemanticChunk(
                text=text,
                start_index=sentences[0].start_index,
                end_index=sentences[-1].end_index,
                token_count=token_count,
                sentences=sentences,
            )
        elif self.return_type == "texts":
            return "".join(sent.text for sent in sentences)
        else:
            raise ValueError("Invalid return_type. Must be either 'chunks' or 'texts'.")
        
    def _compute_chunk_embedding(self, chunk: SemanticChunk) -> np.ndarray:
        """Compute embedding for a chunk by weighted average of its sentence embeddings."""
        return self._compute_group_embedding(chunk.sentences)

    def _split_chunks(
        self, sentence_groups: List[List[Sentence]]
    ) -> List[SemanticChunk]:
        """Split sentence groups into chunks that respect chunk_size.

        Args:
            sentence_groups: List of semantically coherent sentence groups

        Returns:
            List of SemanticChunk objects

        """
        chunks = []

        for group in sentence_groups:
            current_chunk_sentences = []
            current_tokens = 0

            for sentence in group:
                test_tokens = (
                    current_tokens
                    + sentence.token_count
                    + (1 if current_chunk_sentences else 0)
                )

                if test_tokens <= self.chunk_size:
                    # Add to current chunk
                    current_chunk_sentences.append(sentence)
                    current_tokens = test_tokens
                else:
                    # Create chunk if we have sentences
                    if current_chunk_sentences:
                        chunks.append(self._create_chunk(current_chunk_sentences))

                    # Start new chunk with current sentence
                    current_chunk_sentences = [sentence]
                    current_tokens = sentence.token_count

            # Create final chunk for this group
            if current_chunk_sentences:
                chunks.append(self._create_chunk(current_chunk_sentences))

        # Post-process: Combine small chunks with semantically closest neighbor
        if len(chunks) > 1:
            i = 0
            while i < len(chunks):
                if chunks[i].token_count < self.min_chunk_size:
                    current_embedding = self._compute_chunk_embedding(chunks[i])
                    best_similarity = -1
                    best_merge_index = None
                    merge_with_next = False

                    # Try next chunk
                    if i + 1 < len(chunks):
                        next_chunk = chunks[i + 1]
                        combined_tokens = chunks[i].token_count + next_chunk.token_count
                        if combined_tokens <= self.chunk_size:
                            next_embedding = self._compute_chunk_embedding(next_chunk)
                            next_similarity = self._get_semantic_similarity(
                                current_embedding, next_embedding
                            )
                            if next_similarity > best_similarity:
                                best_similarity = next_similarity
                                best_merge_index = i + 1
                                merge_with_next = True

                    # Try previous chunk
                    if i > 0:
                        prev_chunk = chunks[i - 1]
                        combined_tokens = chunks[i].token_count + prev_chunk.token_count
                        if combined_tokens <= self.chunk_size:
                            prev_embedding = self._compute_chunk_embedding(prev_chunk)
                            prev_similarity = self._get_semantic_similarity(
                                current_embedding, prev_embedding
                            )
                            if prev_similarity > best_similarity:
                                best_similarity = prev_similarity
                                best_merge_index = i - 1
                                merge_with_next = False

                    # Merge with the most semantically similar neighbor
                    if best_merge_index is not None:
                        if merge_with_next:
                            # Merge with next chunk
                            merged_sentences = chunks[i].sentences + chunks[best_merge_index].sentences
                            chunks[i] = self._create_chunk(merged_sentences)
                            chunks.pop(best_merge_index)
                        else:
                            # Merge with previous chunk
                            merged_sentences = chunks[best_merge_index].sentences + chunks[i].sentences
                            chunks[best_merge_index] = self._create_chunk(merged_sentences)
                            chunks.pop(i)
                            i -= 1
                        continue
                i += 1

        return chunks

    def chunk(self, text: str) -> List[SemanticChunk]:
        """Split text into semantically coherent chunks using two-pass approach.

        First groups sentences by semantic similarity, then splits groups to respect
        chunk_size while maintaining sentence boundaries.

        Args:
            text: Input text to be chunked

        Returns:
            List of SemanticChunk objects containing the chunked text and metadata

        """
        if not text.strip():
            return []

        if self.verbose:
            print("\nStarting chunking process...")
            print(f"Input text length: {len(text)} characters")

        # Prepare sentences with precomputed information
        sentences = self._prepare_sentences(text)
        if self.verbose:
            print(f"Split into {len(sentences)} sentences")

        if len(sentences) <= self.min_sentences:
            if self.verbose:
                print(f"Text has {len(sentences)} sentences (<= min_sentences={self.min_sentences})")
                print("Returning single chunk")
            return [self._create_chunk(sentences)]

        # Validate sentence sizes
        if self.verbose:
            print("\nValidating sentence sizes...")
        self._validate_sentence_sizes(sentences)
        
        # Calculate similarity threshold
        if self.verbose:
            print("\nCalculating similarity threshold...")
        self.similarity_threshold = self._calculate_similarity_threshold(sentences)
        if self.verbose:
            print(f"Using similarity threshold: {self.similarity_threshold}")

        # First pass: Group sentences by semantic similarity
        if self.verbose:
            print("\nGrouping sentences by semantic similarity...")
        sentence_groups = self._group_sentences(sentences)
        if self.verbose:
            print(f"Created {len(sentence_groups)} initial groups")
            for i, group in enumerate(sentence_groups):
                print(f"Group {i}: {len(group)} sentences - {' '.join(s.text for s in group)}")

        # Second pass: Split groups into size-appropriate chunks
        if self.verbose:
            print("\nSplitting groups into size-appropriate chunks...")
        chunks = self._split_chunks(sentence_groups)
        if self.verbose:
            print(f"Final number of chunks: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i}: {chunk.token_count} tokens - {chunk.text!r}")

        return chunks

    def __repr__(self) -> str:
        """Return a string representation of the SemanticChunker."""
        return (
            f"SemanticChunker(embedding_model={self.embedding_model}, "
            f"mode={self.mode}, "
            f"chunk_size={self.chunk_size}, "
            f"threshold={self.threshold}, "
            f"similarity_window={self.similarity_window}, "
            f"min_sentences={self.min_sentences}, "
            f"min_chunk_size={self.min_chunk_size})"
        )
