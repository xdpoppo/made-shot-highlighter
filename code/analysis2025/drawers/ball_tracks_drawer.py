from .utils import draw_traingle

class BallTracksDrawer:
    """Draws a marker at the ball's position on each frame."""

    def __init__(self):
        self.ball_pointer_color = (0, 255, 0)

    def draw(self, video_frames, tracks):
        """Draws a triangle pointer over the ball's bbox in every frame."""
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            ball_dict = tracks[frame_num]

            for _, ball in ball_dict.items():
                if ball["bbox"] is None:
                    continue
                frame = draw_traingle(frame, ball["bbox"],self.ball_pointer_color)

            output_video_frames.append(frame)
            
        return output_video_frames