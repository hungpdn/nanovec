package errors

import "errors"

var (
	// ErrDimMismatch is returned when the vector dimension does not match the index configuration.
	ErrDimMismatch = errors.New("dimension mismatch")

	// ErrQueryDimMismatch is returned when the query vector dimension does not match the index dimension.
	ErrQueryDimMismatch = errors.New("query dimension mismatch")

	// ErrIDAlreadyExists can be added if you want a standardized error for duplicates
	ErrIDAlreadyExists = errors.New("id already exists")

	// ErrBatchSizeMismatch
	ErrBatchSizeMismatch = errors.New("batch size mismatch")
)
