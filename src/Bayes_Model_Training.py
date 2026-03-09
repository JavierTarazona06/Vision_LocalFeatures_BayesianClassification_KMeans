import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from features import extract_feature_image, flatten_feature_image, list_feature_spaces


LABEL_SKIN = 1
LABEL_NON_SKIN = -1


class ROISelector:
    def __init__(self):
        self.start = None
        self.end = None
        self.current_roi = None
        self.is_drawing = False

    def clear(self):
        self.start = None
        self.end = None
        self.current_roi = None
        self.is_drawing = False

    def callback(self, event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.end = (x, y)
            self.current_roi = None
            self.is_drawing = True
            return

        if event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            self.end = (x, y)
            return

        if event == cv2.EVENT_LBUTTONUP and self.is_drawing:
            self.end = (x, y)
            self.is_drawing = False
            self.current_roi = self._normalized_roi()

    def preview_roi(self):
        if self.start is None or self.end is None:
            return None
        return self._normalized_roi()

    def _normalized_roi(self):
        if self.start is None or self.end is None:
            return None
        x0 = min(self.start[0], self.end[0])
        x1 = max(self.start[0], self.end[0])
        y0 = min(self.start[1], self.end[1])
        y1 = max(self.start[1], self.end[1])
        if x1 == x0 or y1 == y0:
            return None
        return x0, y0, x1, y1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Bayesian training for skin vs non-skin pixel classification."
    )
    parser.add_argument("train_image", help="Training image path.")
    parser.add_argument(
        "--test-image",
        default=None,
        help="Optional test image path. Default: training image.",
    )
    parser.add_argument(
        "--features",
        choices=list_feature_spaces(),
        default="ycrcb",
        help="Feature space used for training and prediction.",
    )
    parser.add_argument(
        "--model",
        choices=["qda", "gnb"],
        default="qda",
        help="Classifier model.",
    )
    parser.add_argument(
        "--decision",
        choices=["map", "ml"],
        default="map",
        help="Decision criterion. map uses empirical priors, ml uses uniform priors.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/bayes",
        help="Base output directory for predictions and metadata.",
    )
    return parser.parse_args()


def load_color_image(image_path):
    img_bgr = cv2.imread(str(image_path), -1)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image '{image_path}'.")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel color image.")
    return img_bgr


def annotate_training_samples(train_bgr, train_features):
    selector = ROISelector()
    window_name = "Training image"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, selector.callback)

    feature_batches = []
    label_batches = []
    roi_count = {"skin": 0, "non_skin": 0}
    pixel_count = {"skin": 0, "non_skin": 0}
    label_key_map = {
        ord("p"): ("skin", LABEL_SKIN),
        ord("n"): ("non_skin", LABEL_NON_SKIN),
    }

    print("Instructions: draw ROI with mouse, press 'p' for skin, 'n' for non-skin.")
    print("Press 'r' to reset current ROI and 'q' when annotation is done.")

    while True:
        display = train_bgr.copy()
        roi_to_draw = selector.preview_roi()
        if roi_to_draw is not None:
            x0, y0, x1, y1 = roi_to_draw
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)

        cv2.putText(
            display,
            "Draw ROI | p:skin n:non_skin r:reset q:train",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, display)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            selector.clear()
            continue

        if key not in label_key_map:
            continue

        if selector.current_roi is None:
            print("No ROI selected. Draw a non-empty rectangle first.")
            continue

        class_name, label_value = label_key_map[key]
        x0, y0, x1, y1 = selector.current_roi
        roi_features = train_features[y0:y1, x0:x1]
        if roi_features.size == 0:
            print("Empty ROI ignored.")
            selector.clear()
            continue

        batch_features = roi_features.reshape(-1, roi_features.shape[2])
        batch_labels = np.full((batch_features.shape[0],), label_value, dtype=np.int32)
        feature_batches.append(batch_features)
        label_batches.append(batch_labels)
        roi_count[class_name] += 1
        pixel_count[class_name] += batch_features.shape[0]
        print(
            f"Recorded {class_name} ROI: ({x0},{y0})-({x1},{y1})"
            f" with {batch_features.shape[0]} pixels."
        )
        selector.clear()

    cv2.destroyWindow(window_name)

    if not feature_batches:
        raise RuntimeError("No training samples were annotated.")

    X_train = np.concatenate(feature_batches, axis=0)
    y_train = np.concatenate(label_batches, axis=0)
    if LABEL_SKIN not in y_train or LABEL_NON_SKIN not in y_train:
        raise RuntimeError(
            "Need both classes before training. Add at least one 'p' ROI and one 'n' ROI."
        )
    return X_train, y_train, roi_count, pixel_count


def make_classifier(model_name, decision):
    priors = None if decision == "map" else [0.5, 0.5]
    if model_name == "qda":
        return QuadraticDiscriminantAnalysis(priors=priors, reg_param=1e-6)
    return GaussianNB(priors=priors)


def predict_mask_and_probability(classifier, feature_image):
    height, width = feature_image.shape[:2]
    flat_features = flatten_feature_image(feature_image)
    flat_labels = classifier.predict(flat_features)
    mask = np.where(flat_labels.reshape(height, width) == LABEL_SKIN, 255, 0).astype(np.uint8)

    prob_map_u8 = None
    if hasattr(classifier, "predict_proba"):
        class_list = list(classifier.classes_)
        if LABEL_SKIN in class_list:
            skin_index = class_list.index(LABEL_SKIN)
            probabilities = classifier.predict_proba(flat_features)[:, skin_index]
            prob_map_u8 = np.clip(probabilities.reshape(height, width) * 255.0, 0, 255).astype(
                np.uint8
            )
    return mask, prob_map_u8


def build_overlay(image_bgr, mask):
    color_layer = image_bgr.copy()
    color_layer[mask == 255] = (40, 180, 255)
    return cv2.addWeighted(image_bgr, 0.7, color_layer, 0.3, 0.0)


def save_outputs(
    args,
    run_dir,
    train_path,
    test_path,
    mask,
    overlay,
    probability_map,
    feature_names,
    roi_count,
    pixel_count,
    X_train,
    y_train,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(run_dir / "mask.png"), mask)
    cv2.imwrite(str(run_dir / "overlay.png"), overlay)
    if probability_map is not None:
        cv2.imwrite(str(run_dir / "skin_probability.png"), probability_map)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_image": str(train_path),
        "test_image": str(test_path),
        "features": args.features,
        "feature_names": list(feature_names),
        "model": args.model,
        "decision": args.decision,
        "class_labels": {"skin": LABEL_SKIN, "non_skin": LABEL_NON_SKIN},
        "roi_count": roi_count,
        "pixel_count": pixel_count,
        "train_samples": int(X_train.shape[0]),
        "feature_dimension": int(X_train.shape[1]),
        "train_label_distribution": {
            "skin": int(np.sum(y_train == LABEL_SKIN)),
            "non_skin": int(np.sum(y_train == LABEL_NON_SKIN)),
        },
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def main():
    args = parse_args()
    train_path = Path(args.train_image)
    test_path = Path(args.test_image) if args.test_image is not None else train_path

    train_bgr = load_color_image(train_path)
    print(
        "Training image:",
        train_bgr.shape[0],
        "rows x",
        train_bgr.shape[1],
        "cols x",
        train_bgr.shape[2],
        "channels",
    )
    train_feature_image, feature_names = extract_feature_image(train_bgr, args.features)
    X_train, y_train, roi_count, pixel_count = annotate_training_samples(
        train_bgr, train_feature_image
    )

    classifier = make_classifier(args.model, args.decision)
    classifier.fit(X_train, y_train)

    test_bgr = load_color_image(test_path)
    test_feature_image, _ = extract_feature_image(test_bgr, args.features)
    mask, probability_map = predict_mask_and_probability(classifier, test_feature_image)
    overlay = build_overlay(test_bgr, mask)

    run_dir = (
        Path(args.output_dir)
        / test_path.stem
        / f"{args.features}_{args.model}_{args.decision}"
    )
    save_outputs(
        args,
        run_dir,
        train_path,
        test_path,
        mask,
        overlay,
        probability_map,
        feature_names,
        roi_count,
        pixel_count,
        X_train,
        y_train,
    )
    print(f"Outputs saved in: {run_dir}")

    cv2.imshow("Predicted mask", mask)
    cv2.imshow("Skin overlay", overlay)
    if probability_map is not None:
        cv2.imshow("Skin probability", probability_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
