"mn.py"    produces music notes saved as .wav files in notes folder.

"mn1.py"    produces music using pysine library, which takes frequency as input and produces the sound; but lags a lot.

"mn2.py"    multiprocessing was introduced for the first time in this code, but the function of process1 is called in process2, which still makes it lag and dilutes the idea of multiprocessing at the first place.

"mn4.py"    multiprocessing is not used, but new sound generating function (independently tested in "pygame_musicgen.py" is introduced and defined which uses pygame to generate sound by sine wave; but lags a lot.

"mn5.py"    this was improvised version of "mn2.py" and "mn4.py", uses multiprocessing, sends data from one process to other using queue, and "mn4.py's" sound generation method is used but the sine waves overlap and construction and destruction of waves occur.

"mn6.py"    this is the improvised version of "mn5.py", but overlap is reduced to a good extent but noise is there while playing.

"mn7.py"    this is the final version to be demonstrated, it has all gesture features planned, and the music generation is same as previous program

"mnp1.py"    this is based on completely changed base code than all other version, but is worse. (not recommended to work on)

Note: if any change must be done, then refer to "mn7.py" as it is the most refined one.
