// Package mnist provides the MNIST database of handwritten digits: 60000
// training and 10000 test images of 28 by 28 greyscale pixels, each with a
// label from 0 to 9.
//
// The four original idx files are embedded in the package, so nothing is
// downloaded at run time and the data is versioned with the code. That
// costs about 11 MB in any binary that imports this package, which is why
// the data lives in its own module: importers of the framework itself
// never fetch it.
//
// The files are the ones distributed by Yann LeCun, Corinna Cortes and
// Christopher Burges; see the package README for provenance and the
// checksums they were verified against.
package mnist

import (
	"compress/gzip"
	"embed"
	"encoding/binary"
	"fmt"
	"io"
	"sync"
)

//go:embed data/*.gz
var files embed.FS

// Set is one split of the database. Pixels holds N images of Rows by Cols
// bytes each, row-major and without padding, so image i starts at
// i*Rows*Cols. A byte is the ink of one pixel, 0 for background and 255
// for full stroke. Labels holds the digit shown in each image.
type Set struct {
	N      int
	Rows   int
	Cols   int
	Pixels []byte
	Labels []byte
}

// Image returns the pixels of image i as a view into the set. It does not
// copy; writing to the result changes the set.
func (s *Set) Image(i int) []byte {
	n := s.Rows * s.Cols
	return s.Pixels[i*n : (i+1)*n]
}

// Float32 writes image i into dst as values from 0 to 1 and returns dst.
// A nil or short dst is allocated to the right length, so the usual call
// is buf = set.Float32(i, buf) inside a loop.
func (s *Set) Float32(i int, dst []float32) []float32 {
	n := s.Rows * s.Cols
	if cap(dst) < n {
		dst = make([]float32, n)
	}
	dst = dst[:n]
	for j, p := range s.Image(i) {
		dst[j] = float32(p) / 255
	}
	return dst
}

const (
	magicImages = 0x00000803
	magicLabels = 0x00000801
)

// Decode reads one idx image file and its label file, both uncompressed,
// and returns them as a set. It is exported for the reader who wants to
// point the loader at their own copy of the files.
func Decode(images, labels io.Reader) (*Set, error) {
	var hdr struct{ Magic, N, Rows, Cols int32 }
	if err := binary.Read(images, binary.BigEndian, &hdr); err != nil {
		return nil, fmt.Errorf("image header: %w", err)
	}
	if hdr.Magic != magicImages {
		return nil, fmt.Errorf("image magic is %#x, want %#x", hdr.Magic, magicImages)
	}
	if hdr.N <= 0 || hdr.Rows <= 0 || hdr.Cols <= 0 {
		return nil, fmt.Errorf("image header says %d images of %dx%d", hdr.N, hdr.Rows, hdr.Cols)
	}

	var lhdr struct{ Magic, N int32 }
	if err := binary.Read(labels, binary.BigEndian, &lhdr); err != nil {
		return nil, fmt.Errorf("label header: %w", err)
	}
	if lhdr.Magic != magicLabels {
		return nil, fmt.Errorf("label magic is %#x, want %#x", lhdr.Magic, magicLabels)
	}
	if lhdr.N != hdr.N {
		return nil, fmt.Errorf("%d images but %d labels", hdr.N, lhdr.N)
	}

	set := &Set{
		N:      int(hdr.N),
		Rows:   int(hdr.Rows),
		Cols:   int(hdr.Cols),
		Pixels: make([]byte, int(hdr.N)*int(hdr.Rows)*int(hdr.Cols)),
		Labels: make([]byte, int(lhdr.N)),
	}
	if _, err := io.ReadFull(images, set.Pixels); err != nil {
		return nil, fmt.Errorf("image data: %w", err)
	}
	if _, err := io.ReadFull(labels, set.Labels); err != nil {
		return nil, fmt.Errorf("label data: %w", err)
	}
	for i, l := range set.Labels {
		if l > 9 {
			return nil, fmt.Errorf("label %d of image %d is not a digit", l, i)
		}
	}
	return set, nil
}

func open(name string) (io.ReadCloser, error) {
	f, err := files.Open("data/" + name)
	if err != nil {
		return nil, err
	}
	z, err := gzip.NewReader(f)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	return z, nil
}

func load(imageFile, labelFile string) (*Set, error) {
	images, err := open(imageFile)
	if err != nil {
		return nil, err
	}
	defer images.Close()
	labels, err := open(labelFile)
	if err != nil {
		return nil, err
	}
	defer labels.Close()
	return Decode(images, labels)
}

var (
	trainOnce = sync.OnceValues(func() (*Set, error) {
		return load("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
	})
	testOnce = sync.OnceValues(func() (*Set, error) {
		return load("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
	})
)

// Train returns the 60000 training images. The set is decompressed on the
// first call and shared by every later one, so callers must treat it as
// read-only.
func Train() (*Set, error) { return trainOnce() }

// Test returns the 10000 test images, under the same sharing rule as
// Train.
func Test() (*Set, error) { return testOnce() }
