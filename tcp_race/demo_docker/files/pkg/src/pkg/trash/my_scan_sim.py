

"""scan simulator for virtual maps"""

class ScanSimulator2D(object):
    """
    2D LIDAR scan simulator class

    Init params:
        num_beams (int): number of beams in the scan
        fov (float): field of view of the laser scan
        std_dev (float, default=0.01): standard deviation of the generated whitenoise in the scan
        eps (float, default=0.0001): ray tracing iteration termination condition
        theta_dis (int, default=2000): number of steps to discretize the angles between 0 and 2pi for look up
        max_range (float, default=30.0): maximum range of the laser
        seed (int, default=123): seed for random number generator for the whitenoise in scan
    """

    def __init__(self, num_beams, fov, std_dev=0.01, eps=0.0001, theta_dis=2000, max_range=30.0, seed=12345):
        # initialization 
        self.num_beams = num_beams
        self.fov = fov
        self.std_dev = std_dev
        self.eps = eps
        self.theta_dis = theta_dis
        self.max_range = max_range
        self.angle_increment = self.fov / (self.num_beams - 1)
        self.theta_index_increment = theta_dis * self.angle_increment / (2. * np.pi)
        self.orig_c = None
        self.orig_s = None
        self.orig_x = None
        self.orig_y = None
        self.map_height = None
        self.map_width = None
        self.map_resolution = None
        self.dt = None
        
        # white noise generator
        self.rng = np.random.default_rng(seed=seed)

        # precomputing corresponding cosines and sines of the angle array
        theta_arr = np.linspace(0.0, 2*np.pi, num=theta_dis)
        self.sines = np.sin(theta_arr)
        self.cosines = np.cos(theta_arr)
    
    def set_map(self, map_path, map_ext):
        """
        Set the bitmap of the scan simulator by path

            Args:
                map_path (str): path to the map yaml file
                map_ext (str): extension (image type) of the map image

            Returns:
                flag (bool): if image reading and loading is successful
        """
        # TODO: do we open the option to flip the images, and turn rgb into grayscale? or specify the exact requirements in documentation.
        # TODO: throw error if image specification isn't met

        # load map image
        map_img_path = os.path.splitext(map_path)[0] + map_ext
        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)

        # grayscale -> binary
        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 255.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]

        # load map yaml
        with open(map_path, 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata['resolution']
                self.origin = map_metadata['origin']
            except yaml.YAMLError as ex:
                print(ex)

        # calculate map parameters
        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]
        self.orig_s = np.sin(self.origin[2])
        self.orig_c = np.cos(self.origin[2])

        # get the distance transform
        self.dt = get_dt(self.map_img, self.map_resolution)

        return True

    def reset_rng(self, seed):
        """
        Resets the generator object's random sequence by re-constructing the generator.

        Args:
            seed (int): seed for the generator

        Returns:
            None
        """
        self.rng = None
        self.rng = np.random.default_rng(seed=seed)

    def scan(self, pose):
        """
        Perform simulated 2D scan by pose on the given map

            Args:
                pose (numpy.ndarray (3, )): pose of the scan frame (x, y, theta)

            Returns:
                scan (numpy.ndarray (n, )): data array of the laserscan, n=num_beams

            Raises:
                ValueError: when scan is called before a map is set
        """
        if self.map_height is None:
            raise ValueError('Map is not set for scan simulator.')
        scan = get_scan(pose, self.theta_dis, self.fov, self.num_beams, self.theta_index_increment, self.sines, self.cosines, self.eps, self.orig_x, self.orig_y, self.orig_c, self.orig_s, self.map_height, self.map_width, self.map_resolution, self.dt, self.max_range)
        noise = self.rng.normal(0., self.std_dev, size=self.num_beams)
        final_scan = scan + noise
        return final_scan

    def get_increment(self):
        return self.angle_increment
