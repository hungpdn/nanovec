package nanovec

// Config configuration for DB
type Config struct {
	Dimension int // Number of vector dimensions (e.g., 1536 for OpenAI)
}

var DefaultConfig = Config{
	Dimension: 1536,
}
