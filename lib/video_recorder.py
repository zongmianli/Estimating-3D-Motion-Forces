import time
import shutil
import cv2 as cv
import numpy as np
import logging
import argparse
from glob import glob
from os import makedirs, remove
from os.path import join, exists, abspath, dirname, basename, isfile

class VideoRecorder(object):
    def __init__(self, gui, cache_folder, target_folder):
        if gui is not None:
            self.gui = gui
            self.window_id = gui.getWindowID("pinocchio")
        self.cache_folder = cache_folder
        self.cache_path = join(cache_folder, "temp")
        self.target_folder = target_folder
        if not exists(cache_folder):
            makedirs(cache_folder)
        if not exists(target_folder):
            makedirs(target_folder)

    def CaptureScreen(self,
                      save_name,
                      extension='jpg',
                      delay=0.1):
        '''
        it capures the screen using gepetto-viewer's capture function, 
        and save the images to some cache folder. Then it takes an 
        image in the cache folder, move and rename it to target folder, 
        and clean up the cache folder.
        '''
        # capture screen
        self.gui.startCapture(self.window_id, self.cache_path, 'jpg')
        time.sleep(delay)
        self.gui.stopCapture(self.window_id)
        time.sleep(delay)
        # get all copies that are captured, these copies should look identical
        image_paths = sorted(glob(join(self.cache_folder, "*[0-9].jpg")))
        if len(image_paths) == 0:
            raise ValueError("len(image_paths) == 0")
        # move one copy to target_folder, and delete the rest
        save_path = join(self.target_folder, '{0}.{1}'.format(save_name, extension))
        shutil.move(image_paths[0], save_path)
        shutil.rmtree(self.cache_folder)
        makedirs(self.cache_folder)

    def MakeGridImages(self,
                       image_folders,
                       crop_boxes,
                       target_size,
                       margin,
                       save_folder):
        print(target_size)
        print(margin)
        num_sources = len(image_folders)
        if (len(image_folders) != len(crop_boxes)):
            raise ValueError("len(image_folders) != len(crop_boxes)")

        # get the paths to grid images
        grid_image_paths = []
        grid_image_sizes = []
        num_frames = None
        for n in range(num_sources):
            print(image_folders[n])
            image_paths = sorted(glob(join(image_folders[n], "*[0-9].jpg")))
            if len(image_paths) == 0:
                image_paths = sorted(glob(join(image_folders[n], "*[0-9].png")))
            if num_frames is not None and num_frames!=len(image_paths):
                raise ValueError("Image folders contain different number of images! {0} vs {1}".format(num_frames, len(image_paths)))
            num_frames = len(image_paths)
            grid_image_paths.append(image_paths)
            # get frame image sizes
            example_image = cv.imread(image_paths[0])
            h, w = example_image.shape[:2]
            grid_image_sizes.append([h, w])
        # target grid image size
        h_target = target_size[0]
        w_target = target_size[1]
        # get scaling from crop size
        scaling = []
        for n in range(num_sources):
            h_crop = crop_boxes[n][2]
            # w_crop = crop_boxes[n][3]
            scaling.append(h_target/(float)(h_crop))

        for i in range(num_frames):
            frame_img = 255*np.ones((h_target, w_target*num_sources + (num_sources-1)*margin, 3), np.uint8)
            y_offset = 0
            for n in range(num_sources):
                # top-left, top-right, bottom-left, bottom-right
                grid_img = cv.imread(grid_image_paths[n][i])
                y, x, rows, cols = crop_boxes[n]
                grid_img = grid_img[y:(y+rows),x:(x+cols),:]
                grid_img = cv.resize(grid_img, None, fx=scaling[n], fy=scaling[n])
                grid_img_height, grid_img_width = grid_img.shape[:2]
                grid_img_canvas = 255*np.ones((h_target, w_target, 3), np.uint8)
                crop_height = min(grid_img_height, h_target)
                crop_width = min(grid_img_width, w_target)
                grid_img_canvas[:crop_height, :crop_width, :] = grid_img[:crop_height, :crop_width, :].copy()
                frame_img[:,(y_offset+w_target*n):(y_offset+w_target*(n+1)),:] = grid_img_canvas
                y_offset += margin

            cv.imwrite(join(save_folder, "{0:06d}.png".format(i)), frame_img)
            print("image saved to {0:s}".format(join(save_folder, "{0:06d}.png".format(i))))
    
    def MakeGridVideo(self,
                      image_folders,
                      crop_boxes,
                      frame_size,
                      save_path,
                      fps=6):
        '''
        image_folders : list of paths to 4 image folders
        crop_boxes: list of crop info: (i,j,rows,cols)
        frame_size: h,w of the output frame. The function 
        will automatically compute the size of each grid cell 
        image and the scaling factor.
        Example usage:
        from os.path import join, exists, abspath, dirname, basename, isfile
        repo_dir = '/Users/zoli/Work/human-object-contact'
        data_dir = '/Users/zoli/Work/human-object-contact/data/handtool_videos'
        
        action_videos = {"hammer": ["hammer_0010"],
                     "barbell": ["barbell_0008", "barbell_0010"],
                     "scythe": ["scythe_0001", "scythe_0002", "scythe_0003", "scythe_0005", "scythe_0006"],
                     "spade": ["spade_0001", "spade_0002", "spade_0003", "spade_0008"]}
        
        action_videos = {
                     "barbell": ["barbell_0008", "barbell_0010"]}
        
        action_videos = {
                     "spade": ["spade_0001", "spade_0002", "spade_0003", "spade_0008"]}
        
        action_videos = {
                     "scythe": ["scythe_0001", "scythe_0002", "scythe_0003", "scythe_0005", "scythe_0006"]}
        
        for action in action_videos.keys():
            for video_name in action_videos[action]:
                image_folders = [
                    join(data_dir, action, video_name, "vis_reduced", "frames"),
                    join(data_dir, action, video_name, "vis_reduced", "endpoints_openpose"),
                    join(repo_dir, "screen_cap", video_name, "original_view"),
                    join(repo_dir, "screen_cap", video_name, "side_view")
                ]
                frame_size = [800,1200]
                if video_name=="scythe_0002":
                    crop_boxes = [[0,0,400,600],[0,0,400,600],[184,508,630,945],[206,492,526,789]]
                elif video_name=="scythe_0001":
                    crop_boxes = [[0,0,400,600],[0,0,400,600],[156,508,689,1033],[192,380,623,935]]
                elif video_name=="scythe_0003":
                    crop_boxes = [[0,0,400,600],[0,0,400,600],[210,350,706,1059],[248,478,641,962]]
                elif video_name=="scythe_0005":
                    crop_boxes = [[0,0,400,600],[0,0,400,600],[224,560,533,799],[220,388,548,822]]
                elif video_name=="scythe_0006":
                    crop_boxes = [[0,0,400,600],[0,0,400,600],[166,454,672,1008],[198,448,611,917]]
                elif video_name=="spade_0001":
                    crop_boxes = [[0,0,400,600],[0,0,400,600],[234,520,582,873],[258,634,522,783]]
                elif video_name=="spade_0002":
                    crop_boxes = [[0,0,400,600],[0,0,400,600],[210,504,573,860],[152,624,572,858]]
                elif video_name=="spade_0003":
                    crop_boxes = [[0,0,400,600],[0,0,400,600],[172,462,627,941],[162,528,641,962]]
                elif video_name=="spade_0008":
                    crop_boxes = [[0,0,400,600],[0,0,400,600],[158,420,669,1004],[166,524,635,953]]
                elif video_name=="barbell_0010":
                    crop_boxes = [[0,0,400,600],[0,0,400,600],[194,478,673,1009],[184,478,620,930]]
                elif video_name=="barbell_0008":
                    crop_boxes = [[0,0,400,600],[0,0,400,600],[200,478,600,900],[230,488,559,839]]
                elif video_name=="barbell_0003":
                    crop_boxes = [[0,0,400,600],[0,0,400,600],[88,348,816,1224],[136,420,717,1076]]
                elif video_name=="hammer_0010":
                    crop_boxes = [[0,0,400,600],[0,0,400,600],[120,412,739,1109],[138,372,715,1072]]
                save_path = join(repo_dir, "temp", video_name+".mp4")

                from lib.video_recorder import VideoRecorder
                cache_folder = "/Users/zoli/Work/human-object-contact/video_release/cache"
                target_folder = join("/Users/zoli/Work/human-object-contact/video_release_v2", video_name)
                recorder = VideoRecorder(None, cache_folder, target_folder)
                recorder.MakeGridVideo(image_folders,
                            crop_boxes,
                            frame_size,
                            save_path,
                            fps=10)
        '''
        if (len(image_folders) != 4):
            raise ValueError("len(image_folders) != 4")
        if (len(crop_boxes) != 4):
            raise ValueError("len(crop_boxes) != 4")
        # get the paths to grid images
        grid_image_paths = []
        grid_image_sizes = []
        num_frames = None
        for n in range(4):
            print(image_folders[n])
            image_paths = sorted(glob(join(image_folders[n], "*[0-9].jpg")))
            if len(image_paths) == 0:
                image_paths = sorted(glob(join(image_folders[n], "*[0-9].png")))
            if num_frames is not None and num_frames!=len(image_paths):
                raise ValueError("Image folders contain different number of images! {0} vs {1}".format(num_frames, len(image_paths)))
            num_frames = len(image_paths)
            grid_image_paths.append(image_paths)
            # get frame image sizes
            example_image = cv.imread(image_paths[0])
            h, w = example_image.shape[:2]
            grid_image_sizes.append([h, w])
        # target grid image size
        h_target = frame_size[0]/2
        w_target = frame_size[1]/2
        # get scaling from crop size
        scaling = []
        for n in range(4):
            h_crop = crop_boxes[n][2]
            # w_crop = crop_boxes[n][3]
            scaling.append(h_target/(float)(h_crop))
        
        # resize crop patches such that they reach the target height
        # then slice each patch into the corresponding grid
        # if a patch's width < w_target, then fill the missing part by white color
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video = cv.VideoWriter(save_path, fourcc, fps, (frame_size[1], frame_size[0]))
        for i in range(num_frames):
            print('frame #{}'.format(i))
            grid_images = []
            for n in range(4):
                # top-left, top-right, bottom-left, bottom-right
                grid_img = cv.imread(grid_image_paths[n][i])
                y, x, rows, cols = crop_boxes[n]
                grid_img = grid_img[y:(y+rows),x:(x+cols),:]
                grid_img = cv.resize(grid_img, None, fx=scaling[n], fy=scaling[n])
                grid_img_height, grid_img_width = grid_img.shape[:2]
                grid_img_canvas = 255*np.ones((h_target, w_target, 3), np.uint8)
                crop_height = min(grid_img_height, h_target)
                crop_width = min(grid_img_width, w_target)
                grid_img_canvas[:crop_height, :crop_width, :] = grid_img[:crop_height, :crop_width, :].copy()
                grid_images.append(grid_img_canvas)
            frame_img = 255*np.ones((frame_size[0], frame_size[1], 3), np.uint8)
            frame_img[:h_target,:w_target,:] = grid_images[0] # topleft
            frame_img[:h_target,w_target:(2*w_target),:] = grid_images[1] # topright
            frame_img[h_target:(2*h_target),:w_target,:] = grid_images[2] # bottomleft
            frame_img[h_target:(2*h_target),w_target:(2*w_target),:] = grid_images[3] # bottomright
            # Write out frame to video
            video.write(frame_img)

        # Release everything if job is finished
        video.release()
        cv.destroyAllWindows()


# viewer = Viewer()
# cache_folder = "/Users/zoli/Work/human-object-contact/screen_cap/cache"
# target_folder = "/Users/zoli/Work/human-object-contact/screen_cap/target"
# recorder = VideoRecorder(viewer.gui, cache_folder, target_folder)

# for i in range(10):
#     recorder.CaptureScreen("image{0}".format(i))


def make_video_from_image_folder(path_to_image_folder,
                                 save_name,
                                 fps=6,
                                 crop_box=None):
    '''
    This function makes a mp4 video from a given folder containing ordered 
    frame images of the same size. It will crop the frame images if 
    crop_box is provided. The default frame rate is 6 frames per second.
    '''

    image_paths = sorted(glob(join(path_to_image_folder, "*[0-9].jpg")))
    nframes = len(image_paths)

    example_image = cv.imread(image_paths[0])
    fheight, fwidth = example_image.shape[:2]

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter(save_name, fourcc, fps, (fwidth, fheight))
    for i in range(nframes):
        print('frame #{}'.format(i))
        frame_img = cv.imread(image_paths[i])
        canvas = frame_img.copy()
        # Write out frame to video
        video.write(canvas)

    # Release everything if job is finished
    video.release()
    cv.destroyAllWindows()

# def MakeGridVideo(self, image_folders, image_formats, crop_boxes, frame_size, margin, save_path, fps=6):
#     '''
#     The function loads image sequences from multiple image folders, concatenates
#     the images from different folders to make a grid of images, and finally
#     saves the resulting grid image sequence as a video.
#     --
#     image_folders: (list of strings) paths to n image folders (n=1,2,...)
#     image_formats: (list of strings). It should be the same length as image_folders.
#     crop_boxes: list of (i,j,h,w). It should be the same length as image_folders.
#     num_colums: (integer) number of columns of the image grid.
#     frame_width: (float) Width of the output image grid. The size of the grid cells are automatically adjusted.
#     margin: the margins between grid images
#     '''
#     num_folders = len(image_folders)
#     if (num_folders != len(image_formats)):
#         raise ValueError("len(image_folders) != len(image_formats)): {0} vs {1}".format(num_folders, len(image_formats)))
#     if (num_folders != len(crop_boxes)):
#         raise ValueError("len(image_folders) != len(crop_boxes)): {0} vs {1}".format(num_folders, len(crop_boxes)))

#     # Get the paths to source images
#     grid_image_paths = []
#     grid_image_sizes = []
#     num_frames = None
#     for n in range(num_folders):
#         image_format = image_formats[n]
#         image_paths = sorted(glob(join(image_folders[n], "*[0-9].{0}".format(image_format))))
#         # Verify the number of frames
#         if num_frames is not None and num_frames!=len(image_paths):
#             raise ValueError("Image folders contain different number of images!")
#         num_frames = len(image_paths)

#         # Save the image paths
#         grid_image_paths.append(image_paths)

#         # get frame image sizes
#         example_image = cv.imread(image_paths[0])
#         h, w = example_image.shape[:2]
#         grid_image_sizes.append([h, w])

#     # target width of the output grid images
#     w_target = frame_width/(float)(num_columns)

#     # get scaling factors from crop size
#     scaling = []
#     h_targets = []
#     for n in range(num_folders):
#         h_crop = crop_boxes[n][2]
#         w_crop = crop_boxes[n][3]
#         scale = w_target/(float)(w_crop)
#         scaling.append(scale)
#         h_targets.append(scale*(float)(h_crop))

#     # Get the frame_height
#     frame_height = 0.
#     for n in range(0,num_folders,num_columns):
#         h_max = 0.
#         # Add the maximum height of the grid images in the row to frame_height
#         frame_height += max(h_targets[n:(min(n+num_columns, num_folders))])

#     # resize crop patches such that they reach the target height
#     # then slice each patch into the corresponding grid
#     # if a patch's width < w_target, then fill the missing part by white color
#     fourcc = cv.VideoWriter_fourcc(*'mp4v')
#     video = cv.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
#     for i in range(num_frames):
#         print('frame #{}'.format(i))
#         #frame_image = self.MakeGridImage()
#         image_paths = []
#         for n in range(num_folders):
#             image_paths.append(grid_image_paths[n][i])

#         # image_paths, num_columns
        
#         num_images = len(image_paths)
#         for n in range(0, num_images, num_columns):
#             grid_image_paths[n][i]

#         grid_images = []
#         for n in range(4):
#             # top-left, top-right, bottom-left, bottom-right
#             grid_img = cv.imread(grid_image_paths[n][i])
#             y, x, rows, cols = crop_boxes[n]
#             grid_img = grid_img[y:(y+rows),x:(x+cols),:]
#             grid_img = cv.resize(grid_img, None, fx=scaling[n], fy=scaling[n])
#             grid_img_height, grid_img_width = grid_img.shape[:2]
#             grid_img_canvas = 255*np.ones((h_target, w_target, 3), np.uint8)
#             crop_height = min(grid_img_height, h_target)
#             crop_width = min(grid_img_width, w_target)
#             grid_img_canvas[:crop_height, :crop_width, :] = grid_img[:crop_height, :crop_width, :].copy()
#             grid_images.append(grid_img_canvas)
#         frame_img = 255*np.ones((frame_size[0], frame_size[1], 3), np.uint8)
#         frame_img[:h_target,:w_target,:] = grid_images[0] # topleft
#         frame_img[:h_target,w_target:(2*w_target),:] = grid_images[1] # topright
#         frame_img[h_target:(2*h_target),:w_target,:] = grid_images[2] # bottomleft
#         frame_img[h_target:(2*h_target),w_target:(2*w_target),:] = grid_images[3] # bottomright
#         # Write out frame to video
#         video.write(frame_img)
#  # Release everything if job is finished
#     video.release()
#     cv.destroyAllWindows()



if __name__ == "__main__":
    # from os.path import join, exists, abspath, dirname, basename, isfile
    # repo_dir = '/Users/zoli/Work/human-object-contact'
    # data_dir = '/Users/zoli/Work/human-object-contact/data/galo'
    
    # # action_videos = {"kv": ["kv06_PKLP"],
    # #                 "mu": ["mu02_PKLP"],
    # #                 "pu":["pu02_PKLP"],
    # #                 "sv": ["sv05_PKLP"]}
    # action_videos = {"pu": ["pu02_PKLP"], "sv":["sv05_PKLP"]}
    
    # for action in action_videos.keys():
    #     for video_name in action_videos[action]:
    #         image_folders = [
    #             join(data_dir, action, video_name, "vis_reduced", "frames"),
    #             join(data_dir, action, video_name, "vis_reduced", "openpose"),
    #             join(repo_dir, "screen_cap", video_name, "original_view"),
    #             join(repo_dir, "screen_cap", video_name, "side_view")
    #         ]
    #         frame_size = [800,1200]
    #         if video_name=="kv06_PKLP":
    #             crop_boxes = [[56,29,489,687],[56,29,489,687],[282,416,694,1041],[358,566,621,932]]
    #         elif video_name=="mu02_PKLP":
    #             crop_boxes = [[56,29,489,687],[56,29,489,687],[131,464,719,1079],[110,419,724,1086]]
    #         elif video_name=="pu02_PKLP":
    #             crop_boxes = [[56,29,489,687],[56,29,489,687],[182,500,677,1016],[155,461,674,1011]]
    #         elif video_name=="sv05_PKLP":
    #             crop_boxes = [[56,29,489,687],[56,29,489,687],[161, 473, 792, 1188],[242,472,739,1108]]
    #         save_path = join(repo_dir, "temp", video_name+".mp4")

    #         from lib.video_recorder import VideoRecorder
    #         cache_folder = "/Users/zoli/Work/human-object-contact/video_release/cache"
    #         target_folder = join("/Users/zoli/Work/human-object-contact/video_release_v2", video_name)
    #         recorder = VideoRecorder(None, cache_folder, target_folder)
    #         recorder.MakeGridVideo(image_folders,
    #                     crop_boxes,
    #                     frame_size,
    #                     save_path,
    #                     fps=10)
    
    
    from os.path import join, exists, abspath, dirname, basename, isfile
    repo_dir = '/Users/zoli/Work/projects/mfv'
    data_dir = '/Users/zoli/Work/data/handtool-1.0.0/data'
    
    action_videos = {
        "hammer": ['hammer_0006', 'hammer_0007', 'hammer_0010'],
        "barbell": ['barbell_0002', 'barbell_0003', 'barbell_0007', 'barbell_0008', 'barbell_0010'],
        "scythe": ['scythe_0001', 'scythe_0002', 'scythe_0003', 'scythe_0005', 'scythe_0006'],
        "spade": ['spade_0001', 'spade_0002', 'spade_0003', 'spade_0007', 'spade_0008']}



    aspect_ratio = 1.5 # w/h

    for action in action_videos.keys():
        for video_name in action_videos[action]:
            print(video_name)
            image_folders = [
                join(data_dir, action, video_name, "frames"),
                "/Users/zoli/Work/data/handtool-1.0.0/object_2d_endpoints/20190615_openpose+endpoints/{0:s}".format(video_name),
                "/Users/zoli/Work/projects/mfv/screenshots/{0:s}/{1:s}_s7".format("0604_handtool_0", video_name),
                "/Users/zoli/Work/projects/mfv/screenshots/{0:s}/{1:s}_s7".format("0604_handtool_60", video_name)
            ]
            frame_size = [800,1200]


            # ---------------------------------------------------------------------------
            # # With big screen (sumsung or dell?)
            # if video_name=="spade_0002":
            #     j1 = 941
            #     i1 = 418
            #     h1 = 455
            #     j2 = 832
            #     i2 = 411
            #     h2 = 521
            # elif video_name=="barbell_0008":
            #     j1 = 733
            #     i1 = 341
            #     h1 = 662
            #     j2 = 801
            #     i2 = 373
            #     h2 = 604
            # elif video_name=="hammer_0006":
            #     j1 = 319
            #     i1 = 169
            #     h1 = 506
            #     j2 = 338
            #     i2 = 171
            #     h2 = 509
            # elif video_name=="hammer_0010":
            #     j1 = 380
            #     i1 = 145
            #     h1 = 510
            #     j2 = 327
            #     i2 = 145
            #     h2 = 500
            # elif video_name=="scythe_0002":
            #     j1 = 952
            #     i1 = 251
            #     h1 = 688
            #     j2 = 850
            #     i2 = 277
            #     h2 = 694
            # ---------------------------------------------------------------------------

            # ---------------------------------------------------------------------------
            # With Macbook Pro's screen
            if video_name=="spade_0001":
                j1 = 420
                i1 = 194
                h1 = 445
                j2 = 459
                i2 = 219
                h2 = 425
            elif video_name=="spade_0002":
                j1 = 555
                i1 = 244
                h1 = 281
                j2 = 495
                i2 = 223
                h2 = 317
            elif video_name=="spade_0003":
                j1 = 326
                i1 = 118
                h1 = 530
                j2 = 346
                i2 = 157
                h2 = 490
            elif video_name=="spade_0007":
                j1 = 517
                i1 = 232
                h1 = 336
                j2 = 636
                i2 = 233
                h2 = 360
            elif video_name=="spade_0008":
                j1 = 349
                i1 = 172
                h1 = 464
                j2 = 379
                i2 = 161
                h2 = 440
            elif video_name=="hammer_0001":
                j1 = 291
                i1 = 129
                h1 = 550
                j2 = 321
                i2 = 142
                h2 = 506
            elif video_name=="hammer_0003":
                j1 = 248
                i1 = 113
                h1 = 612
                j2 = 246
                i2 = 134
                h2 = 553
            elif video_name=="hammer_0006":
                j1 = 278
                i1 = 136
                h1 = 538
                j2 = 357
                i2 = 167
                h2 = 495
            elif video_name=="hammer_0007":
                j1 = 431
                i1 = 145
                h1 = 526
                j2 = 359
                i2 = 135
                h2 = 564
            elif video_name=="hammer_0010":
                j1 = 343
                i1 = 123
                h1 = 550
                j2 = 352
                i2 = 94
                h2 = 560
            elif video_name=="scythe_0001":
                j1 = 367
                i1 = 122
                h1 = 530
                j2 = 422
                i2 = 115
                h2 = 560
            elif video_name=="scythe_0002":
                j1 = 416
                i1 = 112
                h1 = 480
                j2 = 440
                i2 = 119
                h2 = 460
            elif video_name=="scythe_0003":
                j1 = 314
                i1 = 174
                h1 = 474
                j2 = 371
                i2 = 175
                h2 = 480
            elif video_name=="scythe_0005":
                j1 = 425
                i1 = 161
                h1 = 460
                j2 = 412
                i2 = 138
                h2 = 512
            elif video_name=="scythe_0006":
                j1 = 331
                i1 = 105
                h1 = 530
                j2 = 338
                i2 = 104
                h2 = 550
            elif video_name=="barbell_0002":
                j1 = 382
                i1 = 169
                h1 = 500
                j2 = 374
                i2 = 165
                h2 = 508
            elif video_name=="barbell_0003":
                j1 = 410
                i1 = 156
                h1 = 470
                j2 = 405
                i2 = 140
                h2 = 505
            elif video_name=="barbell_0007":
                j1 = 229
                i1 = 81
                h1 = 620
                j2 = 224
                i2 = 73
                h2 = 616
            elif video_name=="barbell_0008":
                j1 = 372
                i1 = 192
                h1 = 420
                j2 = 398
                i2 = 180
                h2 = 428
            elif video_name=="barbell_0010":
                j1 = 330
                i1 = 118
                h1 = 545
                j2 = 301
                i2 = 134
                h2 = 536
            

            w1 = (int)(h1*aspect_ratio)
            w2 = (int)(h2*aspect_ratio)
            print(w1)
            print(w2)

            crop_boxes = [[0,0,400,600],[0,0,400,600], [i1, j1, h1, w1], [i2, j2, h2, w2]]
            
            save_path = join(repo_dir, "temp", video_name+".mp4")

            from lib.video_recorder import VideoRecorder
            cache_folder = "/Users/zoli/Work/projects/mfv/screenshots/cache"
            target_folder = join("/Users/zoli/Work/projects/mfv/screenshots/video_release", video_name)
            recorder = VideoRecorder(None, cache_folder, target_folder)
            #print(image_folders)
            # recorder.MakeGridVideo(image_folders,
            #             crop_boxes,
            #             frame_size,
            #             save_path,
            #             fps=10)

            target_size = [400, 600]
            margin = 5
            save_folder = join(repo_dir, "temp", video_name)
            if not exists(save_folder):
                makedirs(save_folder)
            recorder.MakeGridImages([image_folders[0], image_folders[2], image_folders[3]],
                       [crop_boxes[0], crop_boxes[2], crop_boxes[3]],
                       target_size,
                       margin,
                       save_folder)

    # from os.path import join, exists, abspath, dirname, basename, isfile
    # repo_dir = '/Users/zoli/Work/human-object-contact'
    # data_dir = '/Users/zoli/Work/human-object-contact/data/galo'
    
    # action_videos = {"kv": ["kv06_PKLP"],
    #                 "mu": ["mu02_PKLP"],
    #                 "pu":["pu02_PKLP"],
    #                 "sv": ["sv05_PKLP"]}
    
    # for action in action_videos.keys():
    #     for video_name in action_videos[action]:
    #         image_folders = [
    #             join(data_dir, action, video_name, "vis_reduced", "frames"),
    #             join(data_dir, action, video_name, "vis_reduced", "openpose"),
    #             join(data_dir, action, video_name, "vis_reduced", "hmr"),
    #             join(data_dir, action, video_name, "vis_reduced", "hmr")
    #         ]
    #         frame_size = [800,1200]
    #         if video_name=="kv06_PKLP":
    #             crop_boxes = [[56,29,489,687],[56,29,489,687],[702,234,341,512],[702,908,341,512]]
    #         elif video_name=="mu02_PKLP":
    #             crop_boxes = [[56,29,489,687],[56,29,489,687],[702,234,341,512],[702,908,341,512]]
    #         elif video_name=="pu02_PKLP":
    #             crop_boxes = [[56,29,489,687],[56,29,489,687],[702,234,341,512],[702,908,341,512]]
    #         elif video_name=="sv05_PKLP":
    #             crop_boxes = [[56,29,489,687],[56,29,489,687],[702,234,341,512],[702,908,341,512]]
    #         save_path = join(repo_dir, "temp", video_name+".mp4")

    #         from lib.video_recorder import VideoRecorder
    #         cache_folder = "/Users/zoli/Work/human-object-contact/video_release/cache"
    #         target_folder = join("/Users/zoli/Work/human-object-contact/video_release_v2", video_name)
    #         recorder = VideoRecorder(None, cache_folder, target_folder)
    #         recorder.MakeGridVideo(image_folders,
    #                     crop_boxes,
    #                     frame_size,
    #                     save_path,
    #                     fps=10)
