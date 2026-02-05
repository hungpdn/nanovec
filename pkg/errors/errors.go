package errors

import "errors"

var (
	// ErrDimMismatch is returned when the vector dimension does not match the index configuration.
	ErrDimMismatch = errors.New("dimension mismatch")

	// ErrQueryDimMismatch is returned when the query vector dimension does not match the index dimension.
	ErrQueryDimMismatch = errors.New("query dimension mismatch")

	// ErrIDAlreadyExists can be added if you want a standardized error for duplicates
	ErrIDAlreadyExists = errors.New("id already exists")

	// ErrBatchSizeMismatch is returned when the number of IDs does not match the number of vectors in a batch operation.
	ErrBatchSizeMismatch = errors.New("batch size mismatch")

	// ErrReadOnly is returned when database is in read-only mode
	ErrReadOnly = errors.New("database is in read-only mode")
)
