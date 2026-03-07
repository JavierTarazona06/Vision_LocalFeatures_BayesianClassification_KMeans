import cv2
import sys
from pathlib import Path
import argparse

def _save_png(path, image):
  path.parent.mkdir(parents=True, exist_ok=True)
  if not cv2.imwrite(str(path), image):
    raise ValueError(f"Impossible d'ecrire l'image: {path}")


def load_image(image_path):
  # On charge l'image une seule fois, puis on reutilise ses conversions.
  img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
  if img_bgr is None:
    raise ValueError(f"Impossible de lire l'image: {image_path}")

  h, w, c = img_bgr.shape
  print("Dimension de l'image :", h, "lignes x", w, "colonnes x", c, "canaux")
  return img_bgr


def save_rgb_components(img_bgr, output_root):
  rgb_dir = output_root / "rgb"
  _save_png(rgb_dir / "original.png", img_bgr)
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  r, g, b = cv2.split(img_rgb)
  _save_png(rgb_dir / "R.png", r)
  _save_png(rgb_dir / "G.png", g)
  _save_png(rgb_dir / "B.png", b)


def save_hsv_components(img_bgr, output_root):
  hsv_dir = output_root / "hsv"
  _save_png(hsv_dir / "original.png", img_bgr)
  img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
  h_channel, s_channel, v_channel = cv2.split(img_hsv)
  _save_png(hsv_dir / "H.png", h_channel)
  _save_png(hsv_dir / "S.png", s_channel)
  _save_png(hsv_dir / "V.png", v_channel)


def save_ycbcr_components(img_bgr, output_root):
  # Chaque espace de couleur est enregistre dans son propre dossier.
  ycbcr_dir = output_root / "ycbcr"
  _save_png(ycbcr_dir / "original.png", img_bgr)
  img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
  y_channel, cr_channel, cb_channel = cv2.split(img_ycbcr)
  _save_png(ycbcr_dir / "Y.png", y_channel)
  _save_png(ycbcr_dir / "Cb.png", cb_channel)
  _save_png(ycbcr_dir / "Cr.png", cr_channel)


def _parse_args():
  parser = argparse.ArgumentParser(
    description="Genere les composantes RGB, HSV et YCbCr d'une image dans des dossiers."
  )
  parser.add_argument(
    "image_in",
    nargs="?",
    help="Chemin de l'image en entrée. Si omis, Parrots.jpg est utilisé.",
  )
  parser.add_argument(
    "-s",
    "--space",
    choices=["rgb", "hsv", "ycbcr", "all"],
    default="all",
    help="Espace de couleur à exporter (par défaut: all).",
  )
  parser.add_argument(
    "-o",
    "--output",
    help="Dossier de sortie. Par défaut: components_output/<nom_image>.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  args = _parse_args()
  default_image = Path(__file__).with_name("Parrots.jpg")
  image_path = args.image_in

  if image_path is None:
    if default_image.exists():
      image_path = str(default_image)
      print(f"Aucune image fournie, utilisation de: {image_path}")
    else:
      print("Usage :", sys.argv[0], "<Image_in>")
      print("Erreur : aucune image fournie et Parrots.jpg introuvable.")
      sys.exit(2)

  try:
    img_bgr = load_image(image_path)
    if args.output:
      output_root = Path(args.output)
    else:
      output_root = Path(__file__).with_name("components_output") / Path(image_path).stem

    if args.space in ("rgb", "all"):
      save_rgb_components(img_bgr, output_root)
    if args.space in ("hsv", "all"):
      save_hsv_components(img_bgr, output_root)
    if args.space in ("ycbcr", "all"):
      save_ycbcr_components(img_bgr, output_root)
    print(f"Export termine dans: {output_root.resolve()}")
  except ValueError as err:
    print(err)
    sys.exit(1)
