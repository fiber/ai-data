// Package shakespeare provides the "tiny Shakespeare" corpus: about
// 1.1 MB of dialogue from Shakespeare's plays as one stream of text,
// speaker names and all. It is the standard toy corpus for
// character-level language models, small enough to train on in minutes
// and structured enough that a model's progress is visible to the eye —
// first letter frequencies, then words, then speaker names followed by a
// colon and a line break.
//
// The text is embedded and verified by checksum, so nothing is
// downloaded at run time.
package shakespeare

import (
	"bytes"
	"compress/gzip"
	_ "embed"
	"io"
	"sync"
)

//go:embed data/tinyshakespeare.txt.gz
var compressed []byte

var once = sync.OnceValues(func() (string, error) {
	z, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		return "", err
	}
	defer z.Close()
	b, err := io.ReadAll(z)
	if err != nil {
		return "", err
	}
	return string(b), nil
})

// Text returns the whole corpus. It is decompressed on the first call and
// shared afterwards; strings are immutable, so callers cannot disturb one
// another.
func Text() (string, error) { return once() }

// Vocabulary returns the distinct bytes of the corpus in ascending order
// and a table from byte to index, with -1 for bytes the corpus never
// uses. A character-level model has one entry per distinct byte — 65 of
// them here, against tens of thousands for a subword vocabulary, which
// is what makes a small model practical on a corpus this size.
func Vocabulary(text string) (symbols []byte, index [256]int) {
	var seen [256]bool
	for i := 0; i < len(text); i++ {
		seen[text[i]] = true
	}
	for c := 0; c < 256; c++ {
		if seen[c] {
			index[c] = len(symbols)
			symbols = append(symbols, byte(c))
		} else {
			index[c] = -1
		}
	}
	return symbols, index
}

// Encode maps text to vocabulary indices using the table from Vocabulary.
func Encode(text string, index [256]int) []int {
	ids := make([]int, len(text))
	for i := 0; i < len(text); i++ {
		ids[i] = index[text[i]]
	}
	return ids
}
