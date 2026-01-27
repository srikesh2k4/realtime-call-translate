FROM node:20-alpine

WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./
RUN npm install

# Copy rest of application
COPY . .

# Build the app
RUN npm run build

EXPOSE 5173

# Use preview mode with host binding
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0"]
