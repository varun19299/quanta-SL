% Create the objects for the color and depth sensors. Device 1 is the color sensor and Device 2 is the depth sensor.
vid = videoinput('kinect', 1);
vid2 = videoinput('kinect', 2);

% Get the source properties for the depth device.
srcDepth = getselectedsource(vid2);

% Set the frames per trigger for both devices to 1.
vid.FramesPerTrigger = 1;
vid2.FramesPerTrigger = 1;

% Set the trigger repeat for both devices to 200, in order to acquire 201 frames from both the color sensor and the depth sensor.
vid.TriggerRepeat = 0;
vid2.TriggerRepeat = 0;

% Configure the camera for manual triggering for both sensors.
triggerconfig([vid vid2],'manual');

% Start both video objects.
start([vid vid2]);

%% Preview
preview([vid vid2]);

%% Trigger the devices, then get the acquired data.
% Trigger 200 times to get the frames.
N = 1;
for i = 1:N
%     input('Press Enter to capture...');
    waitforbuttonpress;
    % Trigger both objects.
    trigger([vid vid2])
    % Get the acquired frames and metadata.
    [imgColor, ts_color, metaData_Color] = getdata(vid);
    [imgDepth, ts_depth, metaData_Depth] = getdata(vid2);
    save(sprintf('kinect_results_%d.mat', i), 'imgColor', 'imgDepth')
    fprintf('Done.\n');
end

%% Stop
stop([vid vid2]);
closepreview([vid vid2]);