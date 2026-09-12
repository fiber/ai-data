# fiber/ai-data

Datasets for the examples and tests of
[github.com/fiber/ai](https://github.com/fiber/ai), in their own module so
that importing the framework never drags the data along.

```
go get github.com/fiber/ai-data
```

Nothing here is downloaded at run time: the files are embedded, versioned
with the code and verified by checksum in the tests. A binary that imports
`mnist` grows by about 11 MB, which is the price of an example that works
offline, on the first try, for ever.

## mnist

The MNIST database of handwritten digits: 60000 training and 10000 test
images, 28 by 28 greyscale pixels, each labelled 0 to 9.

```go
train, err := mnist.Train()
test, err := mnist.Test()

fmt.Println(train.N, train.Rows, train.Cols) // 60000 28 28

buf := make([]float32, 28*28)
for i := range train.N {
    x := train.Float32(i, buf)   // pixels scaled to 0..1
    y := train.Labels[i]         // the digit
    _, _ = x, y
}
```

`Image(i)` returns the raw bytes of one image as a view into the set,
without copying. `Decode(images, labels)` reads the uncompressed idx
format from any pair of readers, for anyone who wants to point the loader
at their own copy of the files.

The sets are decompressed on first use and shared afterwards, so treat
them as read-only.

## Attribution

The data in this repository was created by others and is redistributed
unmodified. See [NOTICE](NOTICE) for the creators, the provenance, the
checksums and the citation to give. In short: MNIST is the work of Yann
LeCun, Corinna Cortes and Christopher J. C. Burges, and the paper to cite
is LeCun, Bottou, Bengio and Haffner, *Gradient-based learning applied to
document recognition*, Proceedings of the IEEE, 1998.

## Licence

The Go code is MIT, see [LICENSE](LICENSE). The datasets are not covered
by it and keep the terms of their own sources; see [NOTICE](NOTICE).
