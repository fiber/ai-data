package mnist_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"strings"
	"testing"

	"github.com/fiber/ai-data/mnist"
)

// The fingerprints and label counts below are of the canonical database.
// They fail if the embedded files are ever replaced by something else,
// which is the one mistake a data module must not make quietly.
func TestSets(t *testing.T) {
	cases := []struct {
		name   string
		load   func() (*mnist.Set, error)
		n      int
		pixels string
		labels string
		counts [10]int
	}{
		{
			name:   "train",
			load:   mnist.Train,
			n:      60000,
			pixels: "741c988805d008ac6e4c904b69001ba184c24b2c540a4ef403f4c71b676cf757",
			labels: "1feba77c54802fa5339a11837ea4b2866434b83314ec45192930f1df69120c13",
			counts: [10]int{5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949},
		},
		{
			name:   "test",
			load:   mnist.Test,
			n:      10000,
			pixels: "6d87418db22cc8025d05968bec9bd5c3932904b23485740db143a061a2c9d161",
			labels: "ddeff807876a9661a1110d45c266c86239a3a1b7d37da0c3716a7a683c852ff5",
			counts: [10]int{980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			set, err := c.load()
			if err != nil {
				t.Fatal(err)
			}
			if set.N != c.n || set.Rows != 28 || set.Cols != 28 {
				t.Fatalf("got %d images of %dx%d, want %d of 28x28", set.N, set.Rows, set.Cols, c.n)
			}
			if len(set.Pixels) != c.n*28*28 || len(set.Labels) != c.n {
				t.Fatalf("got %d pixels and %d labels", len(set.Pixels), len(set.Labels))
			}
			if got := sha256.Sum256(set.Pixels); hex.EncodeToString(got[:]) != c.pixels {
				t.Errorf("pixel fingerprint is %x, want %s", got, c.pixels)
			}
			if got := sha256.Sum256(set.Labels); hex.EncodeToString(got[:]) != c.labels {
				t.Errorf("label fingerprint is %x, want %s", got, c.labels)
			}
			var counts [10]int
			for _, l := range set.Labels {
				counts[l]++
			}
			if counts != c.counts {
				t.Errorf("label counts are %v, want %v", counts, c.counts)
			}
		})
	}
}

// The sets are shared, so a second call must not hand out a second copy.
func TestLoadOnce(t *testing.T) {
	a, err := mnist.Test()
	if err != nil {
		t.Fatal(err)
	}
	b, err := mnist.Test()
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Error("Test returned two different sets")
	}
}

func TestFloat32(t *testing.T) {
	set, err := mnist.Test()
	if err != nil {
		t.Fatal(err)
	}
	// A nil buffer is allocated, a big enough one is reused.
	buf := set.Float32(0, nil)
	if len(buf) != 28*28 {
		t.Fatalf("got %d values, want 784", len(buf))
	}
	again := set.Float32(1, buf)
	if &again[0] != &buf[0] {
		t.Error("Float32 allocated although the buffer was long enough")
	}

	var min, max float32 = 1, 0
	for i := range set.N {
		for _, v := range set.Float32(i, buf) {
			min, max = minf(min, v), maxf(max, v)
		}
	}
	if min != 0 || max != 1 {
		t.Errorf("values run from %v to %v, want 0 to 1", min, max)
	}

	// The scaling is the only thing Float32 does.
	pixels, values := set.Image(7), set.Float32(7, buf)
	for i, p := range pixels {
		if want := float32(p) / 255; values[i] != want {
			t.Fatalf("pixel %d: got %v, want %v", i, values[i], want)
		}
	}
}

func TestImageIsAView(t *testing.T) {
	set, err := mnist.Test()
	if err != nil {
		t.Fatal(err)
	}
	img := set.Image(3)
	if &img[0] != &set.Pixels[3*28*28] {
		t.Error("Image copied instead of viewing")
	}
}

func TestDecodeRejectsBadInput(t *testing.T) {
	images := func(magic, n, rows, cols int32, body int) []byte {
		var b bytes.Buffer
		binary.Write(&b, binary.BigEndian, []int32{magic, n, rows, cols})
		b.Write(make([]byte, body))
		return b.Bytes()
	}
	labels := func(magic, n int32, body int) []byte {
		var b bytes.Buffer
		binary.Write(&b, binary.BigEndian, []int32{magic, n})
		b.Write(make([]byte, body))
		return b.Bytes()
	}

	cases := []struct {
		name           string
		images, labels []byte
		want           string
	}{
		{"empty", nil, nil, "image header"},
		{"wrong image magic", images(0x803+1, 1, 28, 28, 784), labels(0x801, 1, 1), "image magic"},
		{"no images", images(0x803, 0, 28, 28, 0), labels(0x801, 0, 0), "0 images"},
		{"wrong label magic", images(0x803, 1, 28, 28, 784), labels(0x801+1, 1, 1), "label magic"},
		{"count mismatch", images(0x803, 2, 28, 28, 2*784), labels(0x801, 1, 1), "2 images but 1 labels"},
		{"short image data", images(0x803, 1, 28, 28, 100), labels(0x801, 1, 1), "image data"},
		{"short label data", images(0x803, 1, 28, 28, 784), labels(0x801, 1, 0), "label data"},
		{"label out of range", images(0x803, 1, 28, 28, 784), append(labels(0x801, 1, 0), 10), "not a digit"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := mnist.Decode(bytes.NewReader(c.images), bytes.NewReader(c.labels))
			if err == nil {
				t.Fatal("no error")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error is %q, want it to mention %q", err, c.want)
			}
		})
	}
}

func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
