# 📚 API Reference

## Overview

Vision Service provides a comprehensive gRPC API for real-time computer vision operations, specializing in face detection, tracking, and analysis.

---

## 🎯 VisionService

### Methods

#### `AnalyzeFrame`

Analyzes a single image frame and returns detected faces with their properties.

**Endpoint:** `VisionService/AnalyzeFrame`

**Request Message:** `VisionRequest`

```protobuf
message VisionRequest {
    bytes image = 1;  // Binary image data (JPEG, PNG, etc.)
}
```

**Response Message:** `VisionResponse`

```protobuf
message VisionResponse {
    bool person_detected = 1;              // True if any faces detected
    repeated DetectedFace faces = 2;       // Array of detected faces
}
```

**DetectedFace Structure:**

```protobuf
message DetectedFace {
    string id = 1;                // Unique face identifier for tracking
    int32 x = 2;                  // X coordinate of face bounding box
    int32 y = 3;                  // Y coordinate of face bounding box
    int32 width = 4;              // Width of face bounding box
    int32 height = 5;             // Height of face bounding box
    repeated float landmarks = 6;  // 68 facial landmark points (x,y pairs)
    bytes face_image = 7;         // Cropped face image (JPEG encoded)
}
```

#### Example Usage

**Python Client:**

```python
import grpc
import proto.vision_pb2 as vision_pb2
import proto.vision_pb2_grpc as vision_pb2_grpc

# Create gRPC channel
channel = grpc.insecure_channel('localhost:50051')
stub = vision_pb2_grpc.VisionServiceStub(channel)

# Read image file
with open('image.jpg', 'rb') as f:
    image_data = f.read()

# Make request
request = vision_pb2.VisionRequest(image=image_data)
response = stub.AnalyzeFrame(request)

# Process response
print(f"Person detected: {response.person_detected}")
print(f"Number of faces: {len(response.faces)}")

for i, face in enumerate(response.faces):
    print(f"Face {i+1}:")
    print(f"  ID: {face.id}")
    print(f"  Position: ({face.x}, {face.y})")
    print(f"  Size: {face.width}x{face.height}")
    print(f"  Landmarks: {len(face.landmarks)//2} points")

    # Save cropped face
    with open(f'face_{face.id}.jpg', 'wb') as f:
        f.write(face.face_image)
```

**Node.js Client:**

```javascript
const grpc = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");
const fs = require("fs");

// Load proto
const packageDefinition = protoLoader.loadSync("vision.proto");
const visionProto = grpc.loadPackageDefinition(packageDefinition);

// Create client
const client = new visionProto.VisionService(
  "localhost:50051",
  grpc.credentials.createInsecure()
);

// Read image
const imageData = fs.readFileSync("image.jpg");

// Make request
client.AnalyzeFrame({ image: imageData }, (error, response) => {
  if (error) {
    console.error("Error:", error);
    return;
  }

  console.log("Person detected:", response.person_detected);
  console.log("Faces found:", response.faces.length);

  response.faces.forEach((face, index) => {
    console.log(`Face ${index + 1}:`);
    console.log(`  ID: ${face.id}`);
    console.log(`  Position: (${face.x}, ${face.y})`);
    console.log(`  Size: ${face.width}x${face.height}`);
  });
});
```

---

## 🎭 EmotionService Integration

Vision Service automatically forwards detected faces to the Emotion Service for emotion analysis.

### Automatic Forwarding

When faces are detected, the service automatically:

1. Extracts face regions
2. Sends `FaceRequest` to Emotion Service
3. Logs emotion analysis results

**FaceRequest Structure:**

```protobuf
message FaceRequest {
    bytes face_image = 1;         // Cropped face image
    string face_id = 2;           // Face tracking ID
    repeated float landmarks = 3;  // Facial landmarks
}
```

**Expected Emotion Response:**

```protobuf
message EmotionResponse {
    string emotion = 1;       // Detected emotion (happy, sad, angry, etc.)
    float confidence = 2;     // Confidence score (0.0 - 1.0)
}
```

---

## 🗣️ SpeechDetectionService Integration

Similar to emotion analysis, face data is forwarded to Speech Detection Service.

**Expected Speech Response:**

```protobuf
message SpeechResponse {
    bool is_speaking = 1;     // True if person is speaking
    float speaking_time = 2;  // Duration of speech in seconds
}
```

---

## 📊 Response Codes

### Success Responses

- **200 OK**: Request processed successfully
- **Faces detected**: `person_detected = true`, faces array populated
- **No faces detected**: `person_detected = false`, empty faces array

### Error Handling

- **Invalid image data**: Returns `person_detected = false`
- **Processing errors**: Logged and returns empty response
- **Service unavailable**: Connection errors logged

---

## 🔧 Configuration

### gRPC Server Settings

```env
GRPC_HOST=0.0.0.0          # Server bind address
GRPC_PORT=50051            # Server port
GRPC_MAX_WORKERS=10        # Thread pool size
```

### Face Detection Parameters

```env
FACE_MATCH_THRESHOLD=0.7   # Face similarity threshold (0.0-1.0)
FACE_CLEANUP_TIMEOUT=5.0   # Face tracking timeout (seconds)
```

---

## 🎯 Best Practices

### Image Input

- **Supported formats**: JPEG, PNG, BMP, TIFF
- **Recommended size**: 640x480 to 1920x1080
- **File size limit**: 10MB per request
- **Color space**: RGB or BGR

### Performance Optimization

- Use appropriate image resolution (higher = slower processing)
- Batch multiple frames for better throughput
- Consider image compression for network efficiency

### Face Tracking

- Face IDs persist across frames for the same person
- IDs are reset after `FACE_CLEANUP_TIMEOUT` seconds
- Use face IDs to correlate emotion/speech data

---

## 🧪 Testing

### Test Image

```python
# Test with sample image
import base64

# Small test image (1x1 pixel)
test_image = base64.b64decode(
    '/9j/4AAQSkZJRgABAQEAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A'
)

request = vision_pb2.VisionRequest(image=test_image)
response = stub.AnalyzeFrame(request)
```

### Health Check

```python
def health_check():
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = vision_pb2_grpc.VisionServiceStub(channel)

        # Test with minimal image
        response = stub.AnalyzeFrame(
            vision_pb2.VisionRequest(image=b'test'),
            timeout=5.0
        )
        return True
    except grpc.RpcError as e:
        print(f"Health check failed: {e}")
        return False
```

---

## 📈 Monitoring

### Metrics

- Request rate (requests/second)
- Processing latency (milliseconds)
- Face detection rate (faces/frame)
- Error rate (errors/requests)

### Logging

All requests are logged with:

- Timestamp
- Request size
- Processing time
- Number of faces detected
- Error details (if any)

---

**For more information, see the [main documentation](./README.md).**
