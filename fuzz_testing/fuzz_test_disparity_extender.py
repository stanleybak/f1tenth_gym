"""
disparity extender test generation
"""

import math
import time

import numpy as np
import mockrospy as rospy

from fuzz_test_gym import Driver, fuzz_test_gym, F110GymSim

# sensor message types
class LaserScan:
    def __init__(self, ranges):
        self.ranges = ranges

class drive_params:
    def __init__(self, angle=0, velocity=0):
        self.angle = angle
        self.velocity = velocity

class FakeLock:
    """a fake lock"""

    def acquire(self, blocking=True):
        """acquire the lock"""
        
        return True

    def release(self):
        """release the lock"""
        
        return True
    

class DisparityExtenderDriving:
    """ Holds configuration options and some state for controlling the car
    using the simplified obstacle-bloating algorithm I'm calling "disparity
    extending" for now, since I don't know if there's an already established
    term for it. """

    def __init__(self):
        """ Initializes the class with default values, for our 1/10 scale
        traxxas car controlled using a FOCBox. """
        # This is actually "half" of the car width, plus some tolerance.
        # Controls the amount disparities are extended by.
        self.car_width = 0.3 # 0.5 # 1.0
        # This is the difference between two successive LIDAR scan points that
        # can be considered a "disparity". (As a note, at 7m there should be
        # ~0.04m between scan points.)
        self.disparity_threshold = 1.0 # 0.4
        # This is the arc width of the full LIDAR scan data, in degrees
        self.scan_width = 270.0
        # This is the radius to the left or right of the car that must be clear
        # when the car is attempting to turn left or right.
        self.turn_clearance = 0.3
        # This is the maximum steering angle of the car, in degrees.
        self.max_turn_angle = 15.0 #34.0
        # The slowest speed the car will go
        # Good value here is 0.1
        self.min_speed = 0.5
        # The maximum speed the car will go (the absolute max for the motor is
        # 0.5, which is *very* fast). 0.15 is a good max for slow testing.
        self.max_speed = 7.75 #.30
        self.absolute_max_speed = 7.5 # 0.4
        # The forward distance at which the car will go its minimum speed.
        # If there's not enough clearance in front of the car it will stop.
        self.min_distance = 0.35
        # The forward distance over which the car will go its maximum speed.
        # Any distance between this and the minimum scales the speed linearly.
        self.max_distance = 3.0
        # The forward distance over which the car will go its *absolute
        # maximum* speed. This distance indicates there are no obstacles in
        # the near path of the car. Distance between this and the max_distance
        # scales the speed linearly.
        self.no_obstacles_distance = 6.0
        # If forward distance is lower than this, use the alternative method
        # for choosing an angle (where the angle favors a disparity rather than
        # simple distance)
        self.no_u_distance = 4.0
        # These allow us to adjust the angle of points to consider based on how
        # the wheels are oriented.
        self.min_considered_angle = -89.0
        self.max_considered_angle = 89.0
        # We'll use this lock to essentially drop LIDAR packets if we're still
        # processing an older one.
        self.lock = FakeLock()
        self.should_stop = False
        self.total_packets = 1
        self.dropped_packets = 0
        # This is just a field that will hold the LIDAR distances.
        self.lidar_distances = None
        # This contains the LIDAR distances, but only "safe" reachable
        # distances, generated by extending disparities.
        self.masked_disparities = None
        # If we're constraining turns to be in the direction of a disparity,
        # then use this array to find indices of possible directions to move.
        self.possible_disparity_indices = None
        # This field will hold the number of LIDAR samples per degree.
        # Initialized when we first get some LIDAR scan data.
        self.samples_per_degree = 0
        self.pub_drive_param = rospy.Publisher('drive_parameters',
            drive_params, queue_size=5)

    def run(self):
        """ Starts running the car until we want it to stop! """
        
        start_time = time.time()
        rospy.Subscriber('scan', LaserScan, self.lidar_callback)
        rospy.spin()

        # rospy.spin() will run until Ctrl+C or something similar causes it to
        # return. So, at this point we know the script should exit.
        self.should_stop = True
        # Try, with a timeout, to acquire the lock and block the other threads.
        got_lock = False
        for i in range(10):
            got_lock = self.lock.acquire(False)
            if got_lock:
                break
            time.sleep(0.05)
        duration = time.time() - start_time
        msg = drive_params()
        msg.angle = 0.5
        msg.velocity = 0.0
        
        self.pub_drive_param.publish(msg)
        self.pub_drive_param.unregister()
        drop_rate = float(self.dropped_packets) / float(self.total_packets)
        #print( "Done processing. Ran for %fs" % (duration,))
        #print( "Dropped %d/%d (%.02f%%) LIDAR packets." % (self.dropped_packets,
        #    self.total_packets, drop_rate * 100.0))
        if got_lock:
            self.lock.release()

    def find_disparities(self):
        """ Scans each pair of subsequent values, and returns an array of indices
        where the difference between the two values is larger than the given
        threshold. The returned array contains only the index of the first value
        in pairs beyond the threshold. """
        to_return = []
        values = self.lidar_distances
        for i in range(len(values) - 1):
            if abs(values[i] - values[i + 1]) >= self.disparity_threshold:
                to_return.append(i)
        return to_return

    def half_car_samples_at_distance(self, distance):
        """ Returns the number of points in the LIDAR scan that will cover half of
        the width of the car along an arc at the given distance. """
        # This isn't exact, because it's really calculated based on the arc length
        # when it should be calculated based on the straight-line distance.
        # However, for simplicty we can just compensate for it by inflating the
        # "car width" slightly.
        distance_between_samples = math.pi * distance / (180.0 *
            self.samples_per_degree)
        return int(math.ceil(self.car_width / distance_between_samples))

    def extend_disparities(self):
        """ For each disparity in the list of distances, extends the nearest
        value by the car width in whichever direction covers up the more-
        distant points. Puts the resulting values in self.masked_disparities.
        """
        values = self.lidar_distances
        masked_disparities = np.copy(values)
        disparities = self.find_disparities()
        # Keep a list of disparity end points corresponding to safe driving
        # angles directly past a disparity. We will find the longest of these
        # constrained distances in situations where we need to turn towards a
        # disparity.
        self.possible_disparity_indices = []
        #print ("Got %d disparities." % (len(disparities),))
        
        for d in disparities:
            a = values[d]
            b = values[d + 1]
            # If extend_positive is true, then extend the nearer value to
            # higher indices, otherwise extend it to lower indices.
            nearer_value = a
            nearer_index = d
            extend_positive = True
            if b < a:
                extend_positive = False
                nearer_value = b
                nearer_index = d + 1
            samples_to_extend = self.half_car_samples_at_distance(nearer_value)
            current_index = nearer_index
            for i in range(samples_to_extend):
                # Stop trying to "extend" the disparity point if we reach the
                # end of the array.
                if current_index < 0:
                    current_index = 0
                    break
                if current_index >= len(masked_disparities):
                    current_index = len(masked_disparities) - 1
                    break
                # Don't overwrite values if we've already found a nearer point
                if masked_disparities[current_index] > nearer_value:
                    masked_disparities[current_index] = nearer_value
                # Finally, move left or right depending on the direction of the
                # disparity.
                if extend_positive:
                    current_index += 1
                else:
                    current_index -= 1
            self.possible_disparity_indices.append(current_index)
        self.masked_disparities = masked_disparities

    def angle_from_index(self, i):
        """ Returns the angle, in degrees, corresponding to index i in the
        LIDAR samples. """
        min_angle = -(self.scan_width / 2.0)
        return min_angle + (float(i) / self.samples_per_degree)

    def index_from_angle(self, i):
        center_index = self.scan_width * (self.samples_per_degree / 2)
        return center_index + int(i * float(self.samples_per_degree))

    def find_widest_disparity_index(self):
        """ Returns the index of the distance corresponding to the "widest"
        disparity that we can safely target. """
        masked_disparities = self.masked_disparities
        # Keep this at 0.1 so that we won't identify noise as a disparity
        max_disparity = 0.1
        max_disparity_index = None
        for d in self.possible_disparity_indices:
            # Ignore disparities that are behind the car.
            angle = self.angle_from_index(d)
            if (angle < self.min_considered_angle) or (angle >
                self.max_considered_angle):
                continue
            angle = d * self.samples_per_degree
            distance = masked_disparities[d]
            prev = distance
            after = distance
            # The disparity must have been extended from one of the two
            # directions, so we can calculate the distance of the disparity by
            # checking the distance between the points on either side of the
            # index (note that something on the endpoint won't matter here
            # either. The inequalities are just for bounds checking, if either
            # one is outside the array, then we already know the disparity was
            # extended from a different direction.
            if (d - 1) > 0:
                prev = masked_disparities[d - 1]
            if (d + 1) < len(masked_disparities):
                after = masked_disparities[d + 1]
            difference = abs(prev - after)
            if difference > max_disparity:
                max_disparity = difference
                max_disparity_index = d
        return max_disparity_index

    def find_new_angle(self):
        """ Returns the angle of the farthest possible distance that can be reached
        in a direct line without bumping into edges. Returns the distance in meters
        and the angle in degrees. """
        self.extend_disparities()
        limited_values = self.masked_disparities
        max_distance = -1.0e10
        angle = 0.0
        # Constrain the arc of possible angles we consider.
        min_sample_index = self.index_from_angle(self.min_considered_angle)
        max_sample_index = self.index_from_angle(self.max_considered_angle)
        limited_values = limited_values[int(min_sample_index):int(max_sample_index)]

        for i in range(len(limited_values)):
            distance = limited_values[i]
            if distance > max_distance:
                angle = self.min_considered_angle + float(i) / self.samples_per_degree
                max_distance = distance
        return distance, angle

    def scale_speed_linearly(self, speed_low, speed_high, distance,
                             distance_low, distance_high):
        """ Scales the speed linearly in [speed_low, speed_high] based on the
        distance value, relative to the range [distance_low, distance_high]. """
        distance_range = distance_high - distance_low
        ratio = (distance - distance_low) / distance_range

        speed_range = speed_high - speed_low
        return speed_low + (speed_range * ratio)

    def duty_cycle_from_distance(self, distance):
        """ Takes a forward distance and returns a duty cycle value to set the
        car's velocity. Fairly unprincipled, basically just scales the speed
        directly based on distance, and stops if the car is blocked. """
        if distance <= self.min_distance:
            return 0.0
        if distance >= self.no_obstacles_distance:
            return self.absolute_max_speed
        if distance >= self.max_distance:
            return self.scale_speed_linearly(self.max_speed, self.absolute_max_speed,
                                             distance, self.max_distance,
                                             self.no_obstacles_distance)
        return self.scale_speed_linearly(self.min_speed, self.max_speed, distance,
                                         self.min_distance, self.max_distance)

    def degrees_to_steering_percentage(self, degrees):
        """ Returns a steering "percentage" value between 0.0 (left) and 1.0
        (right) that is as close as possible to the requested degrees. The car's
        wheels can't turn more than max_angle in either direction. """
        max_angle = self.max_turn_angle
        if degrees > max_angle:
            return math.radians(max_angle)
        if degrees < -max_angle:
            return math.radians(-max_angle)
        # # This maps degrees from -max_angle to +max_angle to values from 0 to 1.
        # #   (degrees - min_angle) / (max_angle - min_angle)
        # # = (degrees - (-max_angle)) / (max_angle - (-max_angle))
        # # = (degrees + max_angle) / (max_angle * 2)
        # return 1.0 - ((degrees + max_angle) / (2 * max_angle))
        return math.radians(degrees)

    def adjust_angle_for_car_side(self, target_angle):
        """ Takes the target steering angle, the distances from the LIDAR, and the
        angle covered by the LIDAR distances. Basically, this function attempts to
        keep the car from cutting corners too close to the wall. In short, it will
        make the car go straight if it's currently turning right and about to hit
        the right side of the car, or turning left or about to hit the left side 
        f the car. """
        scan_width = self.scan_width
        car_tolerance = self.turn_clearance
        distances = self.lidar_distances
        turning_left = target_angle > 0.0
        # Get the portion of the LIDAR samples facing sideways and backwards on
        # the side of the car in the direction of the turn.
        samples_per_degree = float(len(distances)) / scan_width
        number_of_back_degrees = (scan_width / 2.0) - 90.0
        needed_sample_count = int(number_of_back_degrees * samples_per_degree)
        side_samples = []
        if turning_left:
            side_samples = distances[len(distances) - needed_sample_count:]
        else:
            side_samples = distances[:needed_sample_count]
        # Finally, just make sure no point in the backwards scan is too close.
        # This could definitely be more exact with some proper math.
        for v in side_samples:
            if v <= car_tolerance:
                return 0.0
        return target_angle

    def adjust_angle_to_avoid_uturn(self, target_angle, forward_distance):
        """ When the car's forward distance is small, it can favor turning to
        the side of a wide track. This function attempts to detect when such a
        case may occur and force the steering angle to follow a disparity
        instead. """
        if forward_distance > self.no_u_distance:
            return target_angle
        target_index = self.find_widest_disparity_index()
        if target_index is None:
            return target_angle
        return self.angle_from_index(target_index)

    def update_considered_angle(self, steering_angle):
        actual_angle = steering_angle
        if actual_angle < -self.max_turn_angle:
            actual_angle = -self.max_turn_angle
        if actual_angle > self.max_turn_angle:
            actual_angle = self.max_turn_angle
        self.min_considered_angle = -89.0
        self.max_considered_angle = 89.0
        if actual_angle > 0:
            self.min_considered_angle -= actual_angle
        if actual_angle < 0:
            self.max_considered_angle += actual_angle

    def lidar_callback(self, lidar_data):
        """ This is asynchronously called every time we receive new LIDAR data.
        """

        self.total_packets += 1
        # If the lock is currently locked, then previous LIDAR data is still
        # being processed.
        if not self.lock.acquire(False):
            self.dropped_packets += 1
            return
        # if self.should_stop:
        #     return
        start_time = time.time()
        distances = lidar_data.ranges
        self.lidar_distances = distances
        self.samples_per_degree = float(len(distances)) / self.scan_width
        target_distance, target_angle = self.find_new_angle()
        safe_distances = self.masked_disparities
        forward_distance = safe_distances[int(len(safe_distances) / 2)]
        #target_angle = self.adjust_angle_to_avoid_uturn(target_angle,
        #    forward_distance)
        target_angle = self.adjust_angle_for_car_side(target_angle)
        desired_speed = self.duty_cycle_from_distance(forward_distance)
        self.update_considered_angle(target_angle)
        steering_percentage = self.degrees_to_steering_percentage(target_angle)
        msg = drive_params()
        msg.angle = steering_percentage
        msg.velocity = desired_speed

        #print(f"publishing msg with angle: {msg.angle}, vel: {msg.velocity}")
        
        self.pub_drive_param.publish(msg)
        duration = time.time() - start_time
        self.lock.release()
        #print("(took %.02f ms): Target point %.02fm at %.02f degrees: %.02f" % (
        #    duration * 1000.0, target_distance, target_angle, steering_percentage))

class DisparityExtenderDriver(Driver):
    """Driver for smooth blocking planner"""

    def __init__(self):
        rospy.topics_sub = {} # clear pub/sub topics
        
        self.planner = DisparityExtenderDriving()

        self.driving_params = drive_params(0, 0)
        
        self.pub = rospy.Publisher('scan', LaserScan, queue_size=5)
        self.sub = rospy.Subscriber('drive_parameters', drive_params, self.driving_params_callback)
        
        self.planner.run()

        # there are deepcopy going on, so the callback would the wrong instance if this is global
        # it's not just instance 0 and instance 1, since we have a whole tree of drivers!
        self.topics_sub = rospy.topics_sub
        rospy.topics_sub = None

        #self.counter = 0

    def driving_params_callback(self, params):
        self.driving_params = params
        #self.driving_params.angle = params.angle
        #self.driving_params.velocity = params.velocity

    def plan(self, obs, ego_index):
        """return speed, steer"""

        # use appropriape pubsub
        rospy.topics_sub = self.topics_sub

        self.pub.publish(LaserScan(obs["scans"][ego_index].tolist()))
        # calling publish here will assign driving params

        # ensures it won't get reused elsewhere
        rospy.topics_sub = None

        vel = self.driving_params.velocity
        angle = self.driving_params.angle

        #if ego_index == 1: # slow opponent car
        #    self.counter += 1

        #    if self.counter > 700:
        #        print("slowing")
        #        vel *= 0.75 #1.0

        return vel, angle

def main():
    'main entry point'

    load_progress_from_file = True
    nominal = False
    single_car = False
    use_rrt = True
    max_nodes = 2048

    #F110GymSim.obs_limits[1] = [-2, 2]

    fuzz_test_gym(DisparityExtenderDriver, use_lidar=True, render_on=True, nominal=nominal, single_car=single_car,
                  load_progress_from_file=load_progress_from_file, use_rrt=use_rrt, max_nodes=max_nodes)

if __name__ == "__main__":
    main()
