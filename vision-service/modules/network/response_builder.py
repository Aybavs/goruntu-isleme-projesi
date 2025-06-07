"""
gRPC response nesnelerini oluşturan modül
"""

import proto.vision_pb2 as vision_pb2
from ..config.logger_config import setup_logger

# Logger'ı yapılandır
logger = setup_logger()


class ResponseBuilder:
    """gRPC response nesnelerini oluşturan sınıf"""
    
    @staticmethod
    def create_vision_response(processed_faces):
        """
        İşlenmiş yüz verilerinden VisionResponse oluşturur
        
        Args:
            processed_faces: İşlenmiş yüz verileri listesi
            
        Returns:
            vision_pb2.VisionResponse: gRPC response nesnesi
        """
        try:
            # Ana yanıtı oluştur
            response = vision_pb2.VisionResponse(
                person_detected=len(processed_faces) > 0
            )
            
            # Her işlenmiş yüzü yanıta ekle
            for face_data in processed_faces:
                try:
                    detected_face = ResponseBuilder._create_detected_face(face_data)
                    if detected_face:
                        response.faces.append(detected_face)
                except Exception as face_error:
                    logger.error(f"DetectedFace oluşturma hatası: {str(face_error)}")
                    continue
            
            return response
            
        except Exception as e:
            logger.error(f"VisionResponse oluşturma hatası: {str(e)}")
            return vision_pb2.VisionResponse(person_detected=False)
    
    @staticmethod
    def _create_detected_face(face_data):
        """
        Yüz verisinden DetectedFace nesnesi oluşturur
        
        Args:
            face_data: Yüz verisi dictionary'si
            
        Returns:
            vision_pb2.DetectedFace: DetectedFace nesnesi
        """
        try:
            # DetectedFace nesnesini oluştur
            detected_face = vision_pb2.DetectedFace()
            detected_face.id = face_data['id']
            detected_face.x = face_data['x']
            detected_face.y = face_data['y']
            detected_face.width = face_data['width']
            detected_face.height = face_data['height']
            
            # Yüz landmark'larını ekle
            if face_data['landmarks'] is not None:
                detected_face.landmarks.extend([
                    float(coord) for point in face_data['landmarks'] for coord in point
                ])
            
            # Yüz görüntüsünü ekle
            if face_data['face_image'] is not None:
                detected_face.face_image = face_data['face_image']
            
            return detected_face
            
        except Exception as e:
            logger.error(f"DetectedFace oluşturma detay hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_face_request(detected_face):
        """
        DetectedFace'den FaceRequest oluşturur
        
        Args:
            detected_face: vision_pb2.DetectedFace nesnesi
            
        Returns:
            vision_pb2.FaceRequest: FaceRequest nesnesi
        """
        try:
            face_request = vision_pb2.FaceRequest(
                face_image=detected_face.face_image,
                face_id=detected_face.id,
                landmarks=detected_face.landmarks
            )
            return face_request
            
        except Exception as e:
            logger.error(f"FaceRequest oluşturma hatası: {str(e)}")
            return None
