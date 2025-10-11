# server_ws.py (B)
import io, cv2, json
import numpy as np
# from PIL import Image

from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from utils.pixel_to_cam_coords import detect_objects
from utils.pointcloud_utils import create_roi_pcd_from_depth, calculate_rpy

app = FastAPI()
model_path = "/home/ros/llm_robot/yolo/runs/pose/yolo11n_640_500ep/weights/best.pt"  # 필요시 v8s/m로 교체
model = YOLO(model_path)
target_list = ["cube"]

def decode_color(jpeg_bytes: bytes):
    """OpenCV만 사용 (약간 더 빠름)"""
    arr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Failed to decode color JPEG.")
    return img

def decode_depth_png(png_bytes: bytes):
    arr = np.frombuffer(png_bytes, np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)  # uint16 유지
    if depth is None or depth.dtype != np.uint16:
        raise ValueError("Depth PNG must be uint16 (Z16).")
    return depth

# /ws 경로로 WebSocket 연결을 받겠다는 뜻.
@app.websocket("/ws")

# 연결된 클라이언트와의 양방향 통신 세션이 이 함수 안에서 지속됨.
async def ws_infer(websocket: WebSocket):
    # 클라이언트(A)가 ws://B_IP:8001/ws로 연결을 시도하면,
    # 서버(B)는 accept()를 호출해서 그 연결을 수락함.
    await websocket.accept()
    print("[INFO] cube detect client connected.")
    try:
        # 카메라 정보 수신 (1회)
        init_msg = await websocket.receive_text()
        init = json.loads(init_msg)
        
        if init.get("type") == "camera_info":
            camera_info = init["data"]
            
            camera_matrix = np.array(camera_info["camera_matrix"], dtype=np.float32).reshape(3, 3)
            dist_coeffs = np.array(camera_info["dist_coeffs"], dtype=np.float32).reshape(-1)
            print("[INFO] Received camera matrix from client.")
        else:
            print("[WARN] No camera matrix received. Using default.")
            camera_matrix = np.eye(3)
            dist_coeffs = np.zeros(5)

        # 클라이언트(A)가 계속 프레임을 보내므로, 서버(B)는 무한 루프로 계속 수신·처리
        while True:
            # 1) 헤더(JSON 텍스트)
            # 클라이언트(A)가 헤더를 보내면, 서버(B)는 receive_text()로 헤더를 수신함.
            header_text = await websocket.receive_text()
            header = json.loads(header_text)  # {"frame_id":..., "depth_scale":...}
            if header.get("type") != "frame":
                continue
            
            # 2) color JPEG 수신 (bytes)
            color_jpeg = await websocket.receive_bytes()
            # 3) depth PNG 수신 (bytes)
            depth_png = await websocket.receive_bytes()

            # 4) 디코딩
            bgr = decode_color(color_jpeg)
            depth_u16 = decode_depth_png(depth_png)

            # RGB에서 YOLO로 카메라상 좌표 얻기
            yolo_outputs = detect_objects(bgr, depth_u16,target_list, 
                                  camera_matrix, dist_coeffs, model)
            if not yolo_outputs:
                await websocket.send_text(json.dumps({
                    "frame_id": header["frame_id"],
                    "detections": []
                }))
                continue
            
            # Base상 6-DoF    
            cam_xyz = yolo_outputs[0]["cam_xyz"]
            print(f"cam_xyz: {cam_xyz}")
            
            # PCA로 RPY 계산 
            pcd = create_roi_pcd_from_depth(depth_u16, yolo_outputs, camera_matrix)
            roll_normal, pitch_normal, yaw_pca = calculate_rpy(pcd)

            reply = {
                "frame_id": header["frame_id"],
                "detections": [{
                    "cam_xyz": cam_xyz,
                    "roll_normal": float(roll_normal),
                    "pitch_normal": float(pitch_normal),
                    "yaw_pca": float(yaw_pca)
                }]
            }
            await websocket.send_text(json.dumps(reply))
    except WebSocketDisconnect:
        print("[INFO] Client disconnected.")
    
    except Exception as e:
        # 예기치 못한 에러도 JSON으로 통지 (선택)
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run("cube_detect_api:app", host="0.0.0.0", port=8001, reload=False)
