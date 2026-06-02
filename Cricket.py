from src.Batting.Batting import Batting
from src.Bowling.Bowling import Bowling
import argparse

from src.Bowling.LeftHandedBowling import LeftHandedBowling
from src.Bowling.RightHandedBowling import RightHandedBowling
from src.Batting.LeftHandedBatting import LeftHandedBatting
from src.Batting.RightHandedBatting import RightHandedBatting

VALID_SHOTS = ['auto', 'drive', 'defensive', 'pull', 'sweep', 'cut', 'back-punch']


class Cricket:
    def __init__(self):
        self.batting = Batting()
        self.bowling = Bowling()

    def play(self, option, hand, view, video_path, intended_shot='auto'):
        if option.lower() == str("batting"):
            if hand.lower() == str("left"):
                left_handed = LeftHandedBatting(video_path, intended_shot)
                left_handed.bat(view)
            elif hand.lower() == str("right"):
                right_handed = RightHandedBatting(video_path, intended_shot)
                right_handed.bat(view)

        elif option.lower() == str("bowling"):
            if hand.lower() == str("left"):
                left_handed = LeftHandedBowling()
                left_handed.bowl(view)
            elif hand.lower() == str("right"):
                right_handed = RightHandedBowling()
                right_handed.bowl(view)
        else:
            raise ValueError("Input can only be either Batting or Bowling")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="CricketLytics – pose-based cricket analysis")
    parser.add_argument('-option', "--option", required=True,
                        help="batting or bowling", type=str)
    parser.add_argument('-hand', "--hand", required=True,
                        help="left or right", type=str)
    parser.add_argument('-view', "--view", required=True,
                        help="front or side", type=str)
    parser.add_argument('-video', "--video", required=True,
                        help="path to the recorded video file", type=str)
    parser.add_argument('-shot', "--shot", default='auto',
                        choices=VALID_SHOTS,
                        help=("Shot you are attempting: "
                              "auto | drive | defensive | pull | sweep | cut | back-punch "
                              "(default: auto)"),
                        type=str)
    args = parser.parse_args()
    cricket = Cricket()
    cricket.play(args.option, args.hand, args.view, args.video, args.shot)

