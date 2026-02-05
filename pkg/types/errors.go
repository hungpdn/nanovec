package types

import "errors"

var (
	ErrDimMismatch      = errors.New("dimension mismatch")
	ErrQueryDimMismatch = errors.New("query dimension mismatch")
)
