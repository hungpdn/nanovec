.PHONY: all build test clean bench fmt lint

# Default target
all: fmt test build bench lint clean

# Build the example binaries
build:
	@echo "Building examples..."
	mkdir -p bin
	go build -o bin/example-flat ./examples/basic_flat/main.go
	go build -o bin/example-hnsw ./examples/advanced_hnsw/main.go
	go build -o bin/example-sq8 ./examples/memory_optimized_sq8/main.go
	go build -o bin/example-persist ./examples/persistence/main.go

# Run all tests with race detection
test:
	@echo "Running tests..."
	go test -v -race ./...

# Run benchmarks
bench:
	@echo "Running benchmarks..."
	go test -bench=. -benchmem ./...

# Format code
fmt:
	@echo "Formatting code..."
	go fmt ./...

# Run linter (requires golangci-lint)
lint:
	@echo "Linting code..."
	golangci-lint run --out-format=colored-line-number

# Clean up build artifacts and temp data
clean:
	@echo "Cleaning up..."
	rm -rf bin/
	rm -rf data/
	rm -f *.db *.db.lock *.idx *.store