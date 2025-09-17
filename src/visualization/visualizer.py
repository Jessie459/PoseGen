import math
import pickle

import cv2
import matplotlib
import numpy as np

LINK_RANGE_BODY = (0, 23)  # 23 body links
LINK_RANGE_L_HAND = (23, 43)  # 20 hand links for left hand
LINK_RANGE_R_HAND = (43, 63)  # 20 hand links for right hand

KPT_RANGE_BODY = (0, 24)  # 24 body points
KPT_RANGE_FACE = (24, 92)  # 68 face points
KPT_RANGE_L_HAND = (92, 113)  # 21 points for left hand
KPT_RANGE_R_HAND = (113, 134)  # 21 points for right hand


class CocoWholebodyPoseVisualizer:
    def __init__(
        self,
        body_thickness: int = 4,
        hand_thickness: int = 2,
        body_radius: int = 4,
        hand_radius: int = 4,
        body_kpt_thr: float = 0.5,
        hand_kpt_thr: float = 0.5,
        adaptive_color: bool = False,
    ):
        with open("src/configs/coco_wholebody_openpose.pkl", "rb") as f:
            dataset_meta = pickle.load(f)
        self.dataset_meta = dataset_meta

        self.kpt_colors = dataset_meta["keypoint_colors"]  # (134, 3)
        self.link_colors = dataset_meta["skeleton_link_colors"]  # (63, 3)
        self.skeleton_links = dataset_meta["skeleton_links"]  # 63

        # Default drawing settings
        self.body_thickness = body_thickness
        self.hand_thickness = hand_thickness
        self.body_radius = body_radius
        self.hand_radius = hand_radius
        self.body_kpt_thr = body_kpt_thr
        self.hand_kpt_thr = hand_kpt_thr

        self.adaptive_color = adaptive_color
        self.hand_link_colors = [matplotlib.colors.hsv_to_rgb([i / 20, 1.0, 1.0]) * 255 for i in range(20)]

    def get_color_multiplier(self, score, min_value=0.6, max_value=0.7):
        assert min_value < max_value
        if score >= max_value:
            return 1.0
        if score <= min_value:
            return 0.0
        return (score - min_value) / (max_value - min_value)

    def draw_body(
        self,
        image: np.ndarray,
        points: np.ndarray,
        scores: np.ndarray,
        kpt_thr: float,
        thickness: int,
        radius: int,
        draw_indices: bool = False,
    ):
        for kpts, score in zip(points, scores):
            kpt_colors = self.kpt_colors
            link_colors = self.link_colors

            # Draw skeleton links
            for sk_id, sk in enumerate(self.skeleton_links[LINK_RANGE_BODY[0] : LINK_RANGE_BODY[1]]):
                sk_id += LINK_RANGE_BODY[0]

                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if score[sk[0]] < kpt_thr or score[sk[1]] < kpt_thr:
                    continue

                X = np.array((pos1[0], pos2[0]))
                Y = np.array((pos1[1], pos2[1]))

                color = link_colors[sk_id]
                if not isinstance(color, str):
                    color = tuple(int(c) for c in color)

                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                polygons = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), int(thickness)), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(image, polygons, color)

            # Draw keypoints
            for kpt_idx, kpt in enumerate(kpts[KPT_RANGE_BODY[0] : KPT_RANGE_BODY[1]]):
                _kpt_idx = kpt_idx
                kpt_idx += KPT_RANGE_BODY[0]
                if score[kpt_idx] < kpt_thr:
                    continue
                color = kpt_colors[kpt_idx]
                if not isinstance(color, str):
                    color = tuple(int(c) for c in color)
                cv2.circle(image, (int(kpt[0]), int(kpt[1])), int(radius), color, thickness=-1)
                if draw_indices:
                    cv2.putText(
                        image,
                        str(_kpt_idx),
                        (int(kpt[0]) + 5, int(kpt[1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        thickness=2,
                    )

    def draw_hand(
        self,
        image: np.ndarray,
        input_points: np.ndarray,
        input_scores: np.ndarray,
        hand_side: str,
        kpt_thr: float,
        thickness: int,
        radius: int,
        draw_indices: bool = False,
    ):
        for kpts, scores in zip(input_points, input_scores):
            if hand_side == "left":
                hand_link_range = LINK_RANGE_L_HAND
                hand_kpt_range = KPT_RANGE_L_HAND
            elif hand_side == "right":
                hand_link_range = LINK_RANGE_R_HAND
                hand_kpt_range = KPT_RANGE_R_HAND
            else:
                raise ValueError("invalid `hand_side`")

            # Draw keypoints
            kpt_indices = set()
            for sk in self.skeleton_links[hand_link_range[0] : hand_link_range[1]]:
                if scores[sk[0]] < kpt_thr or scores[sk[1]] < kpt_thr:
                    continue
                kpt_indices.add(sk[0])
                kpt_indices.add(sk[1])

            for kpt_idx, kpt in enumerate(kpts[hand_kpt_range[0] : hand_kpt_range[1]]):
                relative_kpt_idx = kpt_idx
                kpt_idx += hand_kpt_range[0]
                if not self.adaptive_color:
                    if scores[kpt_idx] < kpt_thr:
                        continue
                    if kpt_idx not in kpt_indices:
                        continue
                color = [255, 255, 255]
                if self.adaptive_color:
                    mul = self.get_color_multiplier(scores[kpt_idx])
                    color = [int(c * mul) for c in color]
                cv2.circle(image, (int(kpt[0]), int(kpt[1])), radius, color, thickness=-1)
                if draw_indices:
                    cv2.putText(
                        image,
                        str(relative_kpt_idx),
                        (int(kpt[0]) + 5, int(kpt[1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        [255, 255, 255],
                        thickness=1,
                    )

            # Draw skeleton links
            for sk_id, sk in enumerate(self.skeleton_links[hand_link_range[0] : hand_link_range[1]]):
                sk_id += hand_link_range[0]

                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if not self.adaptive_color:
                    if scores[sk[0]] < kpt_thr or scores[sk[1]] < kpt_thr:
                        continue

                X = np.array((pos1[0], pos2[0]))
                Y = np.array((pos1[1], pos2[1]))

                color = self.hand_link_colors[sk_id - hand_link_range[0]]
                if self.adaptive_color:
                    mul1 = self.get_color_multiplier(scores[sk[0]])
                    mul2 = self.get_color_multiplier(scores[sk[1]])
                    mul = mul1 * mul2
                    color = [int(c * mul) for c in color]

                cv2.line(image, (X[0], Y[0]), (X[1], Y[1]), color, thickness=thickness)

    # def draw_face(
    #     self,
    #     image: np.ndarray,
    #     points: np.ndarray,
    #     scores: np.ndarray,
    #     kpt_thr: float,
    #     radius: int,
    # ):
    #     for kpts, score in zip(points, scores):
    #         kpt_colors = self.kpt_colors
    #         link_colors = self.link_colors

    #         # Draw points
    #         for kid, kpt in enumerate(kpts[KPT_RANGE_FACE[0] : KPT_RANGE_FACE[1]]):
    #             kid += KPT_RANGE_FACE[0]
    #             if score[kid] < kpt_thr:
    #                 continue
    #             color = kpt_colors[kid]
    #             if not isinstance(color, str):
    #                 color = tuple(int(c) for c in color)
    #             cv2.circle(image, (int(kpt[0]), int(kpt[1])), int(radius), color, thickness=-1)

    def __call__(
        self,
        points: np.ndarray,  # (num_persons, 133, 2)
        scores: np.ndarray,  # (num_persons, 133)
        image: np.ndarray = None,  # if image is provided, directly draw on this image
        image_size: tuple[int, int] = None,  # (height, width)
        draw_hand=True,  # True, False, ['left', 'right']
        draw_body=True,
        body_thickness: int = None,
        hand_thickness: int = None,
        body_radius: int = None,
        hand_radius: int = None,
        body_kpt_thr: float = None,
        hand_kpt_thr: float = None,
        draw_indices: bool = False,
    ):
        if image is None:
            assert image_size is not None
            image = np.zeros((*image_size, 3), dtype=np.uint8)

        if points is None or len(points) == 0:
            print("There are no people deteced in the image.")
            return image

        points = np.asarray(points)
        scores = np.asarray(scores) if scores is not None else np.ones(points.shape[:-1])

        # mmpose to openpose
        if points.shape[1] == 133:
            info = np.concatenate((points, scores[:, :, np.newaxis]), axis=2)  # (1, 133, 3)
            neck = np.mean(info[:, [5, 6]], axis=1)
            new_info = np.insert(info, 17, neck, axis=1)  # (1, 134, 3)
            mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
            openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
            new_info[:, openpose_idx] = new_info[:, mmpose_idx]
            points, scores = new_info[:, :, :2], new_info[:, :, 2]

        if draw_body:
            self.draw_body(
                image,
                points,
                scores,
                kpt_thr=body_kpt_thr or self.body_kpt_thr,
                thickness=body_thickness or self.body_thickness,
                radius=body_radius or self.body_radius,
                draw_indices=draw_indices,
            )

        if isinstance(draw_hand, bool):
            hand_sides = ["left", "right"] if draw_hand else []
        else:
            assert isinstance(draw_hand, (tuple, list))
            hand_sides = draw_hand

        for hand_side in hand_sides:
            self.draw_hand(
                image,
                points,
                scores,
                hand_side=hand_side,
                kpt_thr=hand_kpt_thr or self.hand_kpt_thr,
                thickness=hand_thickness or self.hand_thickness,
                radius=hand_radius or self.hand_radius,
                draw_indices=draw_indices,
            )

        return image
