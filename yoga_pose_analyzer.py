import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from ultralytics import YOLO
import os
import tempfile

st.set_page_config(
    page_title="Yoga Pose Analyzer",
    page_icon="🧘‍♀️",
    layout="wide"
)

st.title("Yoga Pose Analyzer")
st.markdown("Upload your yoga pose image to get analysis and corrections!")

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'predicted_class' not in st.session_state:
    st.session_state.predicted_class = None
if 'direction' not in st.session_state:
    st.session_state.direction = None
if 'user_image_path' not in st.session_state:
    st.session_state.user_image_path = None

@st.cache_resource
def load_yoga_model():
    """Load the trained yoga pose classification model"""
    try:
        model = load_model('YogaNet_model_1_1.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_yolo_model():
    """Load YOLO pose detection model"""
    try:
        model = YOLO('yolov8n-pose.pt')
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

reference_poses = {
    "warrior2_left": "reference_poses/warrior2_left.jpg",
    "warrior2_right": "reference_poses/warrior2_right.jpg",
    "tree_left": "reference_poses/tree_left.jpg",
    "tree_right": "reference_poses/tree_right.jpg",
    "plank_left": "reference_poses/plank_left.jpg",
    "plank_right": "reference_poses/plank_right.jpg",
    "goddess_left": "reference_poses/goddess_left.jpg",
    "goddess_right": "reference_poses/goddess_right.jpg",
    "downdog_left": "reference_poses/downdog_left.jpg",
    "downdog_right": "reference_poses/downdog_right.jpg"
}

class_names = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']

def predict_pose(img_path, model):
    """Predict yoga pose from image with CORRECT preprocessing"""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        img_data = x / 255.0
        classes = model.predict(img_data, verbose=0)
        
        predicted_class_index = np.argmax(classes[0])
        predicted_class_name = class_names[predicted_class_index]
        confidence = classes[0][predicted_class_index] * 100
        
        return predicted_class_name, confidence, classes[0]
    except Exception as e:
        st.error(f"Error in pose prediction: {e}")
        return None, None, None

def plot_skeleton(ax, keypoints, connections, color='red', label='', alpha=1.0):
    """Plot skeleton with keypoints and connections"""

    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            ax.plot(x, y, 'o', color=color, markersize=8, alpha=alpha)
    
    for connection in connections:
        pt1, pt2 = connection
        if pt1 < len(keypoints) and pt2 < len(keypoints):
            x1, y1 = keypoints[pt1]
            x2, y2 = keypoints[pt2]
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=alpha)

def align_and_normalize_poses(user_keypoints, ref_keypoints, user_img_shape, ref_img_shape):
    hip_indices = [11, 12]
    shoulder_indices = [5, 6]
    
    def get_hip_center_and_scale(keypoints):
        valid_hips = []
        for idx in hip_indices:
            if idx < len(keypoints) and keypoints[idx, 0] > 0 and keypoints[idx, 1] > 0:
                valid_hips.append(keypoints[idx])
        
        if len(valid_hips) >= 2:
            hip_center = np.mean(valid_hips, axis=0)
            hip_width = np.linalg.norm(valid_hips[1] - valid_hips[0])
            return hip_center, hip_width
        elif len(valid_hips) == 1:
            valid_shoulders = []
            for idx in shoulder_indices:
                if idx < len(keypoints) and keypoints[idx, 0] > 0 and keypoints[idx, 1] > 0:
                    valid_shoulders.append(keypoints[idx])
            
            if valid_shoulders:
                torso_points = valid_hips + valid_shoulders
                torso_center = np.mean(torso_points, axis=0)
                torso_points = np.array(torso_points)
                torso_width = np.max(torso_points[:, 0]) - np.min(torso_points[:, 0])
                torso_height = np.max(torso_points[:, 1]) - np.min(torso_points[:, 1])
                scale = max(torso_width, torso_height)
                return torso_center, scale
            else:
                return valid_hips[0], 100
        else:
            valid_mask = (keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)
            if np.any(valid_mask):
                valid_points = keypoints[valid_mask]
                center = np.mean(valid_points, axis=0)
                bbox_width = np.max(valid_points[:, 0]) - np.min(valid_points[:, 0])
                bbox_height = np.max(valid_points[:, 1]) - np.min(valid_points[:, 1])
                scale = max(bbox_width, bbox_height)
                return center, scale
            else:
                return np.array([0, 0]), 100

    user_center, user_scale = get_hip_center_and_scale(user_keypoints)
    ref_center, ref_scale = get_hip_center_and_scale(ref_keypoints)

    scale_factor = user_scale / ref_scale if ref_scale > 0 else 1.0

    aligned_ref_keypoints = ref_keypoints.copy()
    
    for i in range(len(ref_keypoints)):
        if ref_keypoints[i, 0] > 0 and ref_keypoints[i, 1] > 0:
            aligned_ref_keypoints[i] = ref_keypoints[i] - ref_center
            aligned_ref_keypoints[i] *= scale_factor
            aligned_ref_keypoints[i] += user_center
    
    return user_keypoints, aligned_ref_keypoints

def calculate_pose_angles(keypoints):
    angles = {}

    def angle_between_points(p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3"""
        if np.any([p1[0] == 0, p1[1] == 0, p2[0] == 0, p2[1] == 0, p3[0] == 0, p3[1] == 0]):
            return None
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    if len(keypoints) > 9:
        left_shoulder = keypoints[5]  # Left shoulder
        left_elbow = keypoints[7]     # Left elbow  
        left_wrist = keypoints[9]     # Left wrist
        angles['left_arm'] = angle_between_points(left_shoulder, left_elbow, left_wrist)

    if len(keypoints) > 10:
        right_shoulder = keypoints[6]  # Right shoulder
        right_elbow = keypoints[8]     # Right elbow
        right_wrist = keypoints[10]    # Right wrist
        angles['right_arm'] = angle_between_points(right_shoulder, right_elbow, right_wrist)
    
    if len(keypoints) > 15:
        left_hip = keypoints[11]      # Left hip
        left_knee = keypoints[13]     # Left knee
        left_ankle = keypoints[15]    # Left ankle
        angles['left_leg'] = angle_between_points(left_hip, left_knee, left_ankle)
    
    if len(keypoints) > 16:
        right_hip = keypoints[12]     # Right hip
        right_knee = keypoints[14]    # Right knee
        right_ankle = keypoints[16]   # Right ankle
        angles['right_leg'] = angle_between_points(right_hip, right_knee, right_ankle)
    
    if len(keypoints) > 12:
        left_shoulder = keypoints[5]
        left_hip = keypoints[11]
        right_shoulder = keypoints[6]
        right_hip = keypoints[12]
        
        if not np.any([left_shoulder[0] == 0, left_hip[0] == 0, right_shoulder[0] == 0, right_hip[0] == 0]):
            left_torso_vec = left_hip - left_shoulder
            right_torso_vec = right_hip - right_shoulder
            avg_torso_vec = (left_torso_vec + right_torso_vec) / 2
            vertical_vec = np.array([0, 1])
            
            cos_angle = np.dot(avg_torso_vec, vertical_vec) / (np.linalg.norm(avg_torso_vec) * np.linalg.norm(vertical_vec))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles['torso_tilt'] = np.arccos(cos_angle) * 180 / np.pi
    
    return angles

def provide_pose_feedback(user_angles, ref_angles):
    feedback = []
    threshold = 15  # degrees
    
    for joint, user_angle in user_angles.items():
        if user_angle is not None and joint in ref_angles and ref_angles[joint] is not None:
            diff = abs(user_angle - ref_angles[joint])
            
            if diff > threshold:
                if joint == 'left_arm':
                    if user_angle < ref_angles[joint]:
                        feedback.append("🔸 Straighten your left arm more")
                    else:
                        feedback.append("🔸 Bend your left elbow slightly more")
                        
                elif joint == 'right_arm':
                    if user_angle < ref_angles[joint]:
                        feedback.append("🔸 Straighten your right arm more")
                    else:
                        feedback.append("🔸 Bend your right elbow slightly more")
                        
                elif joint == 'left_leg':
                    if user_angle < ref_angles[joint]:
                        feedback.append("🔸 Straighten your left leg more")
                    else:
                        feedback.append("🔸 Bend your left knee slightly more")
                        
                elif joint == 'right_leg':
                    if user_angle < ref_angles[joint]:
                        feedback.append("🔸 Straighten your right leg more")
                    else:
                        feedback.append("🔸 Bend your right knee slightly more")
                        
                elif joint == 'torso_tilt':
                    if user_angle < ref_angles[joint]:
                        feedback.append("🔸 Keep your torso more upright")
                    else:
                        feedback.append("🔸 Lean forward slightly more")
    
    if not feedback:
        feedback.append("Great job! Your pose alignment is very good!")
    
    return feedback

def compare_poses(user_image_path, reference_image_path, pose_type, yolo_model):
    try:
        user_results = yolo_model(user_image_path)
        ref_results = yolo_model(reference_image_path)
        
        user_img = cv2.imread(user_image_path)
        user_img_rgb = cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB)
        
        ref_img = cv2.imread(reference_image_path)
        ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms  
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            (5, 11), (6, 12)  # Torso
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        axes[0].imshow(user_img_rgb)
        axes[0].set_title(f'Your {pose_type} Pose', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        if user_results[0].keypoints is not None:
            user_keypoints = user_results[0].keypoints.xy[0].cpu().numpy()
            plot_skeleton(axes[0], user_keypoints, connections, color='red')
        
        axes[1].imshow(ref_img_rgb)
        axes[1].set_title(f'Reference {pose_type} Pose', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        if ref_results[0].keypoints is not None:
            ref_keypoints = ref_results[0].keypoints.xy[0].cpu().numpy()
            plot_skeleton(axes[1], ref_keypoints, connections, color='green')
        
        axes[2].imshow(user_img_rgb)
        axes[2].set_title('Pose Comparison & Corrections', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        if user_results[0].keypoints is not None and ref_results[0].keypoints is not None:
            user_kpts = user_results[0].keypoints.xy[0].cpu().numpy()
            ref_kpts = ref_results[0].keypoints.xy[0].cpu().numpy()
            
            aligned_user_kpts, aligned_ref_kpts = align_and_normalize_poses(
                user_kpts, ref_kpts, user_img.shape, ref_img.shape
            )
            
            plot_skeleton(axes[2], aligned_user_kpts, connections, color='red', alpha=0.8)
            plot_skeleton(axes[2], aligned_ref_kpts, connections, color='lime', alpha=0.8)
            
            axes[2].text(10, 30, 'Your Pose', color='red', fontsize=12, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            axes[2].text(10, 60, 'Reference Pose', color='lime', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            user_angles = calculate_pose_angles(aligned_user_kpts)
            ref_angles = calculate_pose_angles(aligned_ref_kpts)
            feedback = provide_pose_feedback(user_angles, ref_angles)
            
            plt.tight_layout()
            return fig, feedback
        
        plt.tight_layout()
        return fig, []
        
    except Exception as e:
        st.error(f"Error in pose comparison: {e}")
        return None

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Your Pose")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of your yoga pose"
    )
    
    if uploaded_file is not None:
        
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Your uploaded image", use_column_width=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image_pil.save(tmp_file.name)
            st.session_state.user_image_path = tmp_file.name
        
        if st.button("Analyze Pose", type="primary"):
            with st.spinner("Analyzing your pose..."):
                yoga_model = load_yoga_model()
                
                if yoga_model is not None:
                    predicted_class, confidence, all_probs = predict_pose(st.session_state.user_image_path, yoga_model)
                    
                    if predicted_class is not None:
                        st.session_state.predicted_class = predicted_class
                        st.session_state.prediction_made = True
                        
                        st.success(f"Detected: **{predicted_class.title()}** (Confidence: {confidence:.1f}%)")
                        
                        st.subheader("All Predictions:")
                        for i, class_name in enumerate(class_names):
                            prob = all_probs[i] * 100
                            st.write(f"**{class_name.title()}**: {prob:.1f}%")

with col2:
    st.header("Pose Analysis")
    
    if st.session_state.prediction_made:
        st.success(f"Detected pose: **{st.session_state.predicted_class.title()}**")
        
        st.subheader("👈👉 Choose Direction")
        st.write("Select the direction/variation of the pose for comparison:")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            if st.button("⬅️ Left", use_container_width=True):
                st.session_state.direction = "left"
                
        with col_right:
            if st.button("➡️ Right", use_container_width=True):
                st.session_state.direction = "right"
        
        if st.session_state.direction:
            st.success(f"Selected direction: **{st.session_state.direction.title()}**")
            
            reference_key = f"{st.session_state.predicted_class}_{st.session_state.direction}"
            
            if reference_key in reference_poses:
                reference_pose_path = reference_poses[reference_key]
                
                st.subheader("Running Pose Comparison...")
                
                if st.button("Compare with Reference Pose", type="primary"):
                    with st.spinner("Analyzing pose differences..."):
                        yolo_model = load_yolo_model()
                        
                        if yolo_model is not None and os.path.exists(reference_pose_path):
                            
                            result = compare_poses(
                                st.session_state.user_image_path,
                                reference_pose_path,
                                st.session_state.predicted_class,
                                yolo_model
                            )
                            
                            if isinstance(result, tuple):
                                comparison_fig, feedback = result
                            else:
                                comparison_fig = result
                                feedback = []
                            
                            if comparison_fig is not None:
                                st.pyplot(comparison_fig)
                                
                                st.subheader("💡 Pose Analysis Results")
                                st.info(f"""
                                **Analysis Complete!**
                                - Your pose: {st.session_state.predicted_class.title()}
                                - Direction: {st.session_state.direction.title()}
                                - Reference used: {reference_key}
                                
                                Check the visualization above to see:
                                - 🔴 Red skeleton: Your current pose
                                - 🟢 Green skeleton: Aligned reference pose
                                """)
                                
                                if feedback:
                                    st.subheader("Personalized Corrections")
                                    for tip in feedback:
                                        st.write(tip)
                                else:
                                    st.success("Excellent! Your pose alignment is spot on!")
                        else:
                            st.error("Could not load pose detection model or reference image not found.")
            else:
                st.error(f"Reference pose not found for {reference_key}")
    else:
        st.info("👆 Please upload an image and click 'Analyze Pose' to get started!")