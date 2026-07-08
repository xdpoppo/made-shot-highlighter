# Made Shot Highlighter

An AI tool that automatically detects and clips made shots from basketball game footage, so players and coaches don't have to manually scrub through full games to find highlight moments.

## The Problem

Building highlight reels or reviewing shooting performance currently means watching entire game recordings and manually marking every made shot. This is slow, doesn't scale as more games get filmed, and makes it hard for players building recruiting tapes or coaches running film sessions to quickly find what matters.

## My Solution

This project takes in raw basketball game video and outputs a folder of short clips — one per detected made shot. The process works in six steps:

1. **Dataset** — Start with a labeled dataset of basketball images (ball, rim, and people already marked with bounding boxes).
2. **Train the detector** — Use this dataset to fine-tune a YOLOv8 object detection model, teaching it to recognize the ball and rim in new video it hasn't seen before.
3. **Run on game footage** — Feed a full game video into the trained model frame by frame. For each frame, it outputs the exact location of the ball and rim.
4. **Track the ball over time** — Follow the ball's position across consecutive frames to build a picture of its trajectory.
5. **Detect a make** — Custom logic checks whether the ball's path passes downward through the rim's location — the pattern that happens when a shot goes in.
6. **Extract the clip** — When a make is detected, automatically cut a short clip (a few seconds before and after) from the source video and save it separately.

Repeating this across an entire game produces a folder of ready-to-review highlight clips, without anyone needing to watch the full game.

Training will run in Google Colab (for free GPU access), with the trained model brought into a local Windsurf environment to build the detection and clipping application itself. Testing will begin on the dataset below before validating on self-filmed game footage.

## Results

*(Update this section once you have real output — e.g. detection accuracy, number of correctly clipped makes out of a test video, or example screenshots in /results.)*

## Repository Structure
- /code — Notebooks and scripts for training the YOLOv8 detector and running the make/miss clip pipeline
- /docs — Research paper draft and pitch deck
- /data — Dataset references (see datasets.md)
- /results — Screenshots, charts, and example output clips

## Dataset

**Basketball People Rim** — 3,984 images labeled with basketball, people, and rim classes, by QueenMary.
https://universe.roboflow.com/queenmary/basketball-poeple-rin/browse

## Project Website



## Contact

Hugo — *hugolinnelson@gmail.com*
