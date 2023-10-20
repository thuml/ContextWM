import json
import os
import pathlib
import time
import collections

import numpy as np

try:
    import wandb
except Exception:
    wandb = None  # dont force wandb dependence


class Logger:

    def __init__(self, step, outputs, multiplier=1):
        self._step = step
        self._outputs = outputs
        self._multiplier = multiplier
        self._last_step = None
        self._last_time = None
        self._metrics = []

    def add(self, mapping, prefix=None):
        step = int(self._step) * self._multiplier
        for name, value in dict(mapping).items():
            name = f"{prefix}_{name}" if prefix else name
            value = np.array(value)
            if len(value.shape) not in (0, 2, 3, 4):
                raise ValueError(
                    f"Shape {value.shape} for name '{name}' cannot be "
                    "interpreted as scalar, image, or video."
                )
            self._metrics.append((step, name, value))

    def scalar(self, name, value):
        self.add({name: value})

    def image(self, name, value):
        self.add({name: value})

    def video(self, name, value):
        self.add({name: value})

    def write(self, fps=False):
        fps and self.scalar("fps", self._compute_fps())
        if not self._metrics:
            return
        for output in self._outputs:
            output(self._metrics)
        self._metrics.clear()

    def _compute_fps(self):
        step = int(self._step) * self._multiplier
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration


class TerminalOutput:
    def __call__(self, summaries):
        # TODO aggregate
        step = max(s for s, _, _, in summaries)
        scalars = {k: float(v) for _, k, v in summaries if len(v.shape) == 0}
        formatted = {k: self._format_value(v) for k, v in scalars.items()}
        print(f"[{step}]", " / ".join(f"{k} {v}" for k, v in formatted.items()))

    def _format_value(self, value):
        if value == 0:
            return "0"
        elif 0.01 < abs(value) < 10000:
            value = f"{value:.2f}"
            value = value.rstrip("0")
            value = value.rstrip("0")
            value = value.rstrip(".")
            return value
        else:
            value = f"{value:.1e}"
            value = value.replace(".0e", "e")
            value = value.replace("+0", "")
            value = value.replace("+", "")
            value = value.replace("-0", "-")
        return value


class JSONLOutput:

    def __init__(self, logdir):
        self._logdir = pathlib.Path(logdir).expanduser()

    def __call__(self, summaries):
        # NOTE: to aggregate values in the same step
        scalar_summaries = collections.defaultdict(lambda: collections.defaultdict(list))
        for step, name, value in summaries:
            if len(value.shape) == 0:
                scalar_summaries[step][name].append(value.item())
        for step in scalar_summaries:
            scalars = {k: np.mean(v) for k, v in scalar_summaries[step].items()}
            with (self._logdir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, **scalars}) + "\n")

        # scalars = {k: float(v) for _, k, v in summaries if len(v.shape) == 0}
        # step = max(s for s, _, _, in summaries)
        # with (self._logdir / "metrics.jsonl").open("a") as f:
        #     f.write(json.dumps({"step": step, **scalars}) + "\n")


class TensorBoardOutput:

    def __init__(self, logdir, fps=20):
        # The TensorFlow summary writer supports file protocols like gs://. We use
        # os.path over pathlib here to preserve those prefixes.
        self._logdir = os.path.expanduser(logdir)
        self._writer = None
        self._fps = fps

    def __call__(self, summaries):
        # TODO to aggregate values in the same step
        import tensorflow as tf

        self._ensure_writer()
        self._writer.set_as_default()
        for step, name, value in summaries:
            if len(value.shape) == 0:
                tf.summary.scalar("scalars/" + name, value, step)
            elif len(value.shape) == 2:
                tf.summary.image(name, value, step)
            elif len(value.shape) == 3:
                tf.summary.image(name, value, step)
            elif len(value.shape) == 4:
                self._video_summary(name, value, step)
        self._writer.flush()

    def _ensure_writer(self):
        if not self._writer:
            import tensorflow as tf

            self._writer = tf.summary.create_file_writer(self._logdir, max_queue=1000)

    def _video_summary(self, name, video, step):
        import tensorflow as tf
        import tensorflow.compat.v1 as tf1

        name = name if isinstance(name, str) else name.decode("utf-8")
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(255 * video, 0, 255).astype(np.uint8)
        try:
            T, H, W, C = video.shape
            summary = tf1.Summary()
            image = tf1.Summary.Image(height=H, width=W, colorspace=C)
            image.encoded_image_string = encode_gif(video, self._fps)
            summary.value.add(tag=name, image=image)
            tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
        except (IOError, OSError) as e:
            print("GIF summaries require ffmpeg in $PATH.", e)
            tf.summary.image(name, video, step)


class TensorBoardOutputPytorch:

    # FIXME image dataformats='CHW' by default

    def __init__(self, logdir, fps=20):
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(str(logdir), max_queue=1000)
        self._fps = fps

    def __call__(self, summaries):
        # NOTE: to aggregate values in the same step
        scalar_summaries = collections.defaultdict(list)
        for step, name, value in summaries:
            if len(value.shape) == 0:
                scalar_summaries[(step, name)].append(value.item())
        for (step, name), value in scalar_summaries.items():
            self._writer.add_scalar("scalars/" + name, np.mean(value), step)

        for step, name, value in summaries:
            # if len(value.shape) == 0:
            #     self._writer.add_scalar("scalars/" + name, value, step)
            # elif len(value.shape) == 2:
            if len(value.shape) == 2:
                self._writer.add_image(name, value, step)
            elif len(value.shape) == 3:
                self._writer.add_image(name, value, step)
            elif len(value.shape) == 4:
                self._video_summary(name, value, step)
                # self._writer.add_video(name,value[None], step,fps=self._fps)
                # vid_tensor: :math:`(N, T, C, H, W)`. The values should lie in [0, 255] for type `uint8` or [0, 1] for type `float`.
                # batch, time, channels, height and width
        self._writer.flush()

    def _video_summary(self, name, video, step):
        # import tensorflow as tf
        # import tensorflow.compat.v1 as tf1
        from tensorboard.compat.proto.summary_pb2 import Summary

        name = name if isinstance(name, str) else name.decode("utf-8")
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(255 * video, 0, 255).astype(np.uint8)
        try:
            T, H, W, C = video.shape
            # summary = tb.RecordWriter()
            image = Summary.Image(height=H, width=W, colorspace=C)
            image.encoded_image_string = encode_gif(video, self._fps)
            self._writer._get_file_writer().add_summary(Summary(value=[Summary.Value(tag=name, image=image)]), step)
            # tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
        except (IOError, OSError) as e:
            print("GIF summaries require ffmpeg in $PATH.", e)
            self._writer.add_image(name, video, step)


def encode_gif(frames, fps):
    from subprocess import Popen, PIPE

    h, w, c = frames[0].shape
    pxfmt = {1: "gray", 3: "rgb24"}[c]
    cmd = " ".join(
        [
            "ffmpeg -y -f rawvideo -vcodec rawvideo",
            f"-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex",
            "[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse",
            f"-r {fps:.02f} -f gif -",
        ]
    )
    proc = Popen(cmd.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tobytes())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError("\n".join([" ".join(cmd), err.decode("utf8")]))
    del proc
    return out


class WandBOutput:

    # FIXME image dataformats='CHW' by default

    def __init__(self, fps=20, **kwargs):
        assert wandb, 'make sure you have wandb installed'

        self._fps = fps
        wandb.init(**kwargs)

    def __call__(self, summaries):
        for step, name, value in summaries:
            if len(value.shape) == 0:
                wandb.log({'scalars/' + name: value}, step=step)
            elif len(value.shape) == 2 or len(value.shape) == 3:
                wandb.log({name: wandb.Image(value)}, step=step)
            elif len(value.shape) == 4:
                name = name if isinstance(name, str) else name.decode('utf-8')
                if np.issubdtype(value.dtype, np.floating):
                    value = np.clip(255 * value, 0, 255).astype(np.uint8)
                value = value.transpose((0, 3, 1, 2))
                wandb.log({name: wandb.Video(value, fps=self._fps, format='mp4')}, step=step)
