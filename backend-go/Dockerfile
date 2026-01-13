FROM golang:1.22-alpine

WORKDIR /app

COPY go.mod ./
RUN go mod download

COPY . .

RUN go build -o server main.go

EXPOSE 8000

CMD ["./server"]
