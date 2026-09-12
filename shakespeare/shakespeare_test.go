package shakespeare_test

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"

	"github.com/fiber/ai-data/shakespeare"
)

// The fingerprint fails if the embedded corpus is ever replaced, which is
// the one mistake a data module must not make quietly.
func TestText(t *testing.T) {
	text, err := shakespeare.Text()
	if err != nil {
		t.Fatal(err)
	}
	if len(text) != 1115394 {
		t.Errorf("corpus is %d bytes, want 1115394", len(text))
	}
	sum := sha256.Sum256([]byte(text))
	const want = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
	if got := hex.EncodeToString(sum[:]); got != want {
		t.Errorf("fingerprint is %s, want %s", got, want)
	}
	if !strings.HasPrefix(text, "First Citizen:\n") {
		t.Errorf("corpus starts %q", text[:min(20, len(text))])
	}
}

func TestTextIsShared(t *testing.T) {
	a, err := shakespeare.Text()
	if err != nil {
		t.Fatal(err)
	}
	b, _ := shakespeare.Text()
	if a != b {
		t.Error("two calls returned different text")
	}
}

func TestVocabularyAndEncode(t *testing.T) {
	symbols, index := shakespeare.Vocabulary("banana\n")
	if got := string(symbols); got != "\nabn" {
		t.Errorf("symbols are %q, want \"\\nabn\" in ascending byte order", got)
	}
	for i, c := range symbols {
		if index[c] != i {
			t.Errorf("index of %q is %d, want %d", c, index[c], i)
		}
	}
	if index['z'] != -1 {
		t.Errorf("an unused byte indexes to %d, want -1", index['z'])
	}
	ids := shakespeare.Encode("banana", index)
	want := []int{2, 1, 3, 1, 3, 1}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("encoded %v, want %v", ids, want)
		}
	}
}

// The corpus itself: 65 distinct bytes is the number every tutorial on
// this dataset quotes, and it is what makes the embedding table small.
func TestCorpusVocabulary(t *testing.T) {
	text, err := shakespeare.Text()
	if err != nil {
		t.Fatal(err)
	}
	symbols, index := shakespeare.Vocabulary(text)
	if len(symbols) != 65 {
		t.Errorf("corpus has %d distinct bytes, want 65", len(symbols))
	}
	ids := shakespeare.Encode(text, index)
	if len(ids) != len(text) {
		t.Fatalf("encoded %d ids for %d bytes", len(ids), len(text))
	}
	for i, id := range ids {
		if id < 0 || id >= len(symbols) {
			t.Fatalf("byte %d encoded to %d, outside the vocabulary", i, id)
		}
		if symbols[id] != text[i] {
			t.Fatalf("byte %d round-tripped to %q, want %q", i, symbols[id], text[i])
		}
	}
}
