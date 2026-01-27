# Multi-stage build for optimized Go backend
FROM golang:1.22-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum* ./
RUN go mod download

# Copy source code
COPY . .

# Build with optimizations
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o server main.go

# Final stage - minimal runtime image
FROM alpine:latest

# Install curl for health checks
RUN apk add --no-cache curl ca-certificates

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/server .

EXPOSE 8000

CMD ["./server"]
