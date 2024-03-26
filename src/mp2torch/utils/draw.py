import cv2
import numpy as np


def draw_bboxes(
    image: np.ndarray,
    bboxes: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    colors: list[tuple[int, int, int]] | None = None,
    bw: int = 1,
) -> np.ndarray:
    """
    bbox: [ymin, xmin, ymax, xmax]
    scores: [score, ...]
    image: [height, width, channel]
    """
    if bboxes is None:
        return image
    bboxes[:, [0, 1]] = np.floor(bboxes[:, [0, 1]])
    bboxes[:, [2, 3]] = np.ceil(bboxes[:, [2, 3]])
    bboxes = bboxes.astype(np.int16)
    if scores is None:
        scores = [None] * bboxes.shape[0]
    if colors is None:
        colors = [None] * bboxes.shape[0]
    for bbox, score, color in zip(bboxes, scores, colors):
        image = draw_bbox(image, bbox=bbox, score=score, color=color, bw=bw)
    return image


def draw_landmarks(
    image: np.ndarray, landmarks: np.ndarray | None = None, bw: int = 1
) -> np.ndarray:
    if landmarks is None:
        return image
    landmarks = landmarks.round()
    landmarks = landmarks.astype(np.int16)
    for landmark in landmarks:
        image = draw_landmark(image, landmark=landmark, bw=bw)
    return image


def draw_landmark(
    image: np.ndarray, landmark: np.ndarray | None = None, bw: int = 1
) -> np.ndarray:
    if landmark is None:
        return image
    for i in range(0, len(landmark), 3):
        x = landmark[i]
        y = landmark[i + 1]
        image[y - 1 : y + 1, x - 1 : x + 1, :] = [0, 255, 255]
    return image


def draw_bbox(
    image: np.ndarray,
    bbox: np.ndarray | None = None,
    score: float | None = None,
    color: tuple[int, int, int] | None = None,
    bw: int = 1,
) -> np.ndarray:
    """
    bbox: [ymin, xmin, ymax, xmax]
    score: float | None
    image: [height, width, channel]
    """
    if bbox is None:
        return image
    if color is None:
        color = (0, 0, 255)
    bbox[[0, 1]] = np.floor(bbox[[0, 1]])
    bbox[[2, 3]] = np.ceil(bbox[[2, 3]])
    bbox = bbox.astype(np.int16)
    image[bbox[0] : bbox[2], bbox[1] - bw : bbox[1] + bw, :] = np.array(
        color
    )  # left vertical
    image[bbox[0] : bbox[2], bbox[3] - bw : bbox[3] + bw, :] = np.array(
        color
    )  # right vertical
    image[bbox[0] - bw : bbox[0] + bw, bbox[1] : bbox[3], :] = np.array(
        color
    )  # top horizontal
    image[bbox[2] - bw : bbox[2] + bw, bbox[1] : bbox[3], :] = np.array(
        color
    )  # bottom horizontal
    if score is not None:
        image = cv2.putText(
            image,
            text=str(score),
            org=(bbox[1], bbox[0]),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return image


def plot_image(
    image: np.ndarray,
    save_to: str,
    bboxes: np.ndarray | None = None,
    landmarks: np.ndarray | None = None,
) -> None:
    image = draw_bboxes(image, bboxes)
    image = draw_landmarks(image, landmarks)
    cv2.imwrite(save_to, image)


def draw_video(
    video_path: str,
    save_to: str,
    bboxes_seq: list[np.ndarray, None] | None = None,
    landmarks_seq: list[np.ndarray, None] | None = None,
    detection_scores_seq: list[np.ndarray, None] | None = None,
    face_ids_seq: list[np.ndarray, None] | None = None,
) -> None:
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

    writer = cv2.VideoWriter(
        save_to,
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )
    if bboxes_seq is None:
        bboxes_seq = []
    if landmarks_seq is None:
        landmarks_seq = []
    bboxes_length = len(bboxes_seq)
    landmarks_length = len(landmarks_seq)
    colors = ((0, 0, 255), (255, 0, 0), (0, 255, 0))
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        if bboxes_length > i + 1:
            if detection_scores_seq is not None:
                frame = draw_bboxes(
                    frame,
                    bboxes=bboxes_seq[i],
                    scores=detection_scores_seq[i],
                    colors=[colors[face_id % 3] for face_id in face_ids_seq[i]]
                    if face_ids_seq is not None
                    else None,
                )
            else:
                frame = draw_bboxes(
                    frame,
                    bboxes=bboxes_seq[i],
                    scores=None,
                    colors=[colors[face_id % 3] for face_id in face_ids_seq[i]]
                    if face_ids_seq is not None
                    else None,
                )
        if landmarks_length > i + 1:
            frame = draw_landmarks(frame, landmarks=landmarks_seq[i])
        writer.write(frame)
    cap.release()
    writer.release()
