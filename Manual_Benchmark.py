#!/usr/bin/env python3

import carla
import pygame
import numpy as np
import time
import math
from Start_server import kill, start_server
from collections import deque
from numba import njit
from global_route_planner import GlobalRoutePlanner
from scipy.spatial import KDTree
import threading

kill()
start_server(2000)
time.sleep(20)


@njit
def my_2d_norm(x):
    return math.sqrt(x[0] ** 2 + x[1] ** 2)


@njit
def coord_transform(vec, anc):
    onm = np.zeros((2, 2))
    det = -anc[0] ** 2 - anc[1] ** 2
    onm[0, 0] = -anc[0]
    onm[1, 1] = anc[0]
    onm[0, 1] = -anc[1]
    onm[1, 0] = -anc[1]
    return onm.dot(vec.transpose()).transpose() / det


@njit
def yaw_to_vector(yaw):
    yaw_radians = math.radians(yaw)
    return np.array([math.cos(yaw_radians), math.sin(yaw_radians)])


def get_route_paper(grp, map):
    def delete_doubles(route):
        last_loc = [(route[0])[0].transform.location.x, (route[0])[0].transform.location.y]
        unique_route = [(route[0])[0]]
        for wp, _ in route[1:]:
            loc = [wp.transform.location.x, wp.transform.location.y]
            if np.linalg.norm(np.array(loc) - np.array(last_loc)) > .3:
                unique_route.append(wp)
                last_loc = loc
        return unique_route

    # Beispiel: Start-Location selbst gemessen oder geschätzt
    start_loc = carla.Location(x=92.6, y=104.8, z=0.5)
    end_loc = carla.Location(x=87.5, y=114.9, z=0.5)

    # Nächste Waypoints auf der Straße finden
    start_wp = map.get_waypoint(start_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    end_wp = map.get_waypoint(end_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

    route = grp.trace_route(start_wp.transform.location, end_wp.transform.location)
    route = delete_doubles(route)

    x, y, z = route[0].transform.location.x, route[0].transform.location.y, route[0].transform.location.z
    spawn_point = carla.Transform(carla.Location(x=x, y=y, z=z + .5), route[0].transform.rotation)

    coords = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in route])
    tree = KDTree(coords)
    return (spawn_point, route), tree


class CarlaBenchmark:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world('Town01_Opt', map_layers=carla.MapLayer.NONE)  # get_world()
        self.world.load_map_layer(carla.MapLayer.All)
        self.map = self.world.get_map()
        self.spawn_points = list(self.map.get_spawn_points())
        self.grp = GlobalRoutePlanner(self.map, sampling_resolution=2)
        self.world.set_weather(carla.WeatherParameters.CloudySunset)
        self.vehicle = None
        self.camera = None
        self.display = None
        self.clock = None

        # Benchmark-spezifische Variablen
        (self.spawn_point, self.route), self.tree = get_route_paper(self.grp,
                                                                    self.map)  # Tupel (spawn_point, waypoint_list)

        self.last_wp_idx = 0
        self.deviations = []
        self.start_time = None
        self.benchmark_active = False
        self.route_completed = False

        # Display-Einstellungen
        self.display_width = 1280
        self.display_height = 720

        # Fahrzeug-Steuerung
        self.control = carla.VehicleControl()
        self.steer_cache = 0.0

    def setup_carla(self, host='localhost', port=2000):
        """Verbindung zu Carla herstellen und Welt initialisieren"""
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = self.client.load_world(map, map_layers=carla.MapLayer.NONE)  # get_world()
            self.world.load_map_layer(carla.MapLayer.All)
            self.map = self.world.get_map()
            self.grp = GlobalRoutePlanner(self.map, sampling_resolution=2)
            self.world.set_weather(carla.WeatherParameters.CloudySunset)
            print("Verbindung zu Carla erfolgreich hergestellt")
            return True
        except Exception as e:
            print(f"Fehler beim Verbinden zu Carla: {e}")
            return False

    def setup_pygame(self):
        """Pygame für die Steuerung initialisieren"""
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (self.display_width, self.display_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Carla Route Benchmark")
        self.clock = pygame.time.Clock()

    def spawn_vehicle(self, spawn_point):
        """Fahrzeug am angegebenen Spawn-Punkt erstellen"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.lincoln.mkz_2020')[0]

        # Fahrzeug spawnen
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Fahrzeug gespawnt at {spawn_point.location}")

        # Kamera für visuelle Rückmeldung (optional)
        self.setup_camera()

    def setup_camera(self):
        """Kamera am Fahrzeug befestigen für visuelle Rückmeldung"""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.display_width))
        camera_bp.set_attribute('image_size_y', str(self.display_height))
        camera_bp.set_attribute('fov', '90')

        # Kamera-Position relativ zum Fahrzeug
        camera_transform = carla.Transform(
            carla.Location(x=-5.5, z=2.8),
            carla.Rotation(pitch=-15)
        )

        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )

        # Kamera-Daten verarbeiten
        self.camera.listen(lambda image: self.process_camera_data(image))

    def process_camera_data(self, image):
        """Kamera-Daten für Pygame-Display verarbeiten"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]  # BGR zu RGB

        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.display.blit(surface, (0, 0))

    def set_route(self, spawn_point, waypoint_list):
        """Route für den Benchmark setzen"""
        self.route = (spawn_point, waypoint_list)
        print(f"Route gesetzt mit {len(waypoint_list)} Waypoints")

    def reset_vehicle_to_start(self):
        """Fahrzeug zurück zur Startposition setzen"""
        if not self.vehicle or not self.spawn_point:
            print("Fehler: Fahrzeug oder Spawn-Point nicht gesetzt!")
            return

        # Benchmark stoppen falls aktiv
        if self.benchmark_active:
            self.benchmark_active = False
            print("Benchmark gestoppt")

        # Fahrzeug zur Startposition teleportieren
        self.vehicle.set_transform(self.spawn_point)

        # Fahrzeug-Geschwindigkeit auf Null setzen
        self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))

        # Benchmark-Variablen zurücksetzen
        self.last_wp_idx = 0
        self.deviations = []
        self.route_completed = False
        self.steer_cache = 0.0

        # Fahrzeugsteuerung zurücksetzen
        reset_control = carla.VehicleControl()
        reset_control.throttle = 0.0
        reset_control.brake = 1.0
        reset_control.steer = 0.0
        reset_control.hand_brake = True
        self.vehicle.apply_control(reset_control)

        print("Fahrzeug zur Startposition zurückgesetzt")

        # Kurz warten und dann Handbremse lösen
        def release_handbrake():
            time.sleep(0.5)
            if self.vehicle:
                normal_control = carla.VehicleControl()
                normal_control.hand_brake = False
                self.vehicle.apply_control(normal_control)

        threading.Thread(target=release_handbrake, daemon=True).start()

    def estimate_dist_to_route(self):
        # veh = self.carla_vehicle
        transform = self.vehicle.get_transform()
        location = np.array([transform.location.x, transform.location.y])
        distances, indices = self.tree.query(location, k=3)
        heading = yaw_to_vector(transform.rotation.yaw)
        filtered_indices = [id for id in indices if abs(id - self.last_wp_idx) <= 10]
        if filtered_indices:
            indices = filtered_indices
        index = indices[0]
        wp = self.route[index]

        vec_dist = 1000
        for id in indices:
            vec = coord_transform(np.array([self.route[id].transform.location.x - location[0],
                                            self.route[id].transform.location.y - location[1]]), heading)
            loc1 = location  # np.array([location.x, location.y])
            loc2 = np.array([self.route[id].transform.location.x, self.route[id].transform.location.y])
            this_dist = my_2d_norm(loc1 - loc2)
            if vec[0] >= 0 and this_dist <= vec_dist:
                index = id
                vec_dist = this_dist
        self.last_wp_idx = index
        wp_before = np.array([self.route[max(indices[0] - 1, 0)].transform.location.x,
                              self.route[max(indices[0] - 1, 0)].transform.location.y])
        wp_nearest = np.array([wp.transform.location.x, wp.transform.location.y])
        wp_after = np.array([self.route[min(indices[0] + 1, len(self.route) - 1)].transform.location.x,
                             self.route[min(indices[0] + 1, len(self.route) - 1)].transform.location.y])
        dist = 1000
        for [wp1, wp2] in [[wp_before, wp_nearest], [wp_nearest, wp_after]]:
            # Vector from w1 to w2
            v = wp2 - wp1
            # Vector from w1 to x
            wp1_to_x = location - wp1
            if my_2d_norm(wp1_to_x) == 0:
                return 0

            if not my_2d_norm(v) == 0:
                t = np.dot(wp1_to_x, v) / np.dot(v, v)
            else:
                t = 0

            # Clamp t to the range [0, 1] to stay within the segment
            t = max(0, min(1, t))

            # Find the closest point on the segment to x
            closest_point = wp1 + t * v

            # Return the distance from x to the closest point
            dist = min(my_2d_norm(location - closest_point), dist)
        for wp in self.route[index:index+30]:
            wp_location = wp.transform.location
            self.world.debug.draw_point(wp_location, .1, color=carla.Color(r=255,g=0,b=0),
                                        life_time=0.3)
        return dist

    def check_route_completion(self):
        """Überprüfen ob die Route abgeschlossen wurde"""
        if not self.vehicle or not self.route:
            return False

        waypoint_list = self.route
        if self.last_wp_idx >= len(waypoint_list) - 3:
            vehicle_location = self.vehicle.get_location()
            final_waypoint = waypoint_list[-1]

            distance = math.sqrt(
                (vehicle_location.x - final_waypoint.transform.location.x) ** 2 +
                (vehicle_location.y - final_waypoint.transform.location.y) ** 2
            )

            return distance < 5.0  # Ziel erreicht wenn näher als 5 Meter

        return False

    def handle_input(self):
        """Benutzer-Input für Fahrzeugsteuerung verarbeiten"""
        keys = pygame.key.get_pressed()

        # Steuerung zurücksetzen
        self.control.throttle = 0.0
        self.control.brake = 0.0
        self.control.steer = 0.0
        self.control.hand_brake = False

        # Pfeiltasten-Steuerung
        if keys[pygame.K_w]:
            self.control.throttle = 1.0
        if keys[pygame.K_s]:
            self.control.brake = 1.0
        if keys[pygame.K_LEFT]:
            self.steer_cache -= 0.05
        if keys[pygame.K_RIGHT]:
            self.steer_cache += 0.05
        if keys[pygame.K_SPACE]:
            self.control.hand_brake = True

        # Lenkung glätten und begrenzen
        self.steer_cache = max(-1.6, min(1.6, self.steer_cache))
        self.control.steer = math.tan(self.steer_cache/2)

        # Lenkung zurückstellen wenn keine Eingabe
        if not (keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]):
            if self.steer_cache > 0.05:
                self.steer_cache -= 0.05
            elif self.steer_cache < -0.05:
                self.steer_cache += 0.05
            else:
                self.steer_cache = 0.0

        # Steuerung anwenden
        self.vehicle.apply_control(self.control)

    def update_benchmark(self):
        """Benchmark-Daten aktualisieren"""
        if not self.benchmark_active:
            return

        # Abweichung zur Route berechnen
        deviation = self.estimate_dist_to_route()
        self.deviations.append(deviation)

        # Route-Completion überprüfen
        if self.check_route_completion():
            self.complete_benchmark()

    def start_benchmark(self):
        """Benchmark starten"""
        if not self.vehicle or not self.route:
            print("Fehler: Fahrzeug oder Route nicht gesetzt!")
            return

        self.benchmark_active = True
        self.start_time = time.time()
        self.deviations = []
        self.last_wp_idx = 0  # Verwende last_wp_idx statt current_waypoint_index
        self.route_completed = False
        print("Benchmark gestartet! Fahre die Route ab.")
        print("Steuerung: Pfeiltasten (Hoch=Gas, Runter=Bremse, Links/Rechts=Lenken), Leertaste=Handbremse")

    def complete_benchmark(self):
        """Benchmark abschließen und Ergebnisse ausgeben"""
        if not self.benchmark_active:
            return

        self.benchmark_active = False
        self.route_completed = True
        end_time = time.time()

        # Ergebnisse berechnen
        total_time = end_time - self.start_time
        mean_deviation = np.mean(self.deviations) if self.deviations else 0.0
        max_deviation = np.max(self.deviations) if self.deviations else 0.0

        # Ergebnisse ausgeben
        print("\n" + "=" * 50)
        print("BENCHMARK ABGESCHLOSSEN!")
        print("=" * 50)
        print(f"Zeit: {total_time:.2f} Sekunden")
        print(f"Mittlere Abweichung: {mean_deviation:.3f} Meter")
        print(f"Maximale Abweichung: {max_deviation:.3f} Meter")
        print(f"Anzahl Messungen: {len(self.deviations)}")
        print("=" * 50)

    def render_speedometer(self):
        """Tacho mit Gas, Bremse, Lenkung und Geschwindigkeit rendern"""
        if not self.vehicle:
            return

        # Fahrzeug-Daten abrufen
        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        # Tacho-Position (unten rechts) - größer gemacht
        tacho_x = self.display_width - 320
        tacho_y = self.display_height - 280
        tacho_width = 300
        tacho_height = 260

        font_large = pygame.font.Font(None, 56)
        font_medium = pygame.font.Font(None, 36)
        font_small = pygame.font.Font(None, 28)

        # Hintergrund für Tacho (größer und mit mehr Padding)
        bg_rect = pygame.Rect(tacho_x - 10, tacho_y - 10, tacho_width, tacho_height)
        pygame.draw.rect(self.display, (20, 20, 20, 220), bg_rect)
        pygame.draw.rect(self.display, (255, 255, 255), bg_rect, 3)

        # Geschwindigkeitsanzeige (groß in der Mitte oben)
        speed_text = font_large.render(f"{speed_kmh:.0f}", True, (255, 255, 255))
        speed_rect = speed_text.get_rect(center=(tacho_x + tacho_width // 2, tacho_y + 40))
        self.display.blit(speed_text, speed_rect)

        kmh_text = font_medium.render("km/h", True, (200, 200, 200))
        kmh_rect = kmh_text.get_rect(center=(tacho_x + tacho_width // 2, tacho_y + 75))
        self.display.blit(kmh_text, kmh_rect)

        # Startposition für Balken (mehr Abstand)
        bar_start_y = tacho_y + 110
        bar_spacing = 50  # Mehr Platz zwischen den Balken
        bar_width = 110
        bar_height = 20
        label_x = tacho_x + 10
        bar_x = tacho_x + 110

        # Gas-Anzeige (grün)
        gas_y = bar_start_y
        gas_label = font_small.render("Gas:", True, (255, 255, 255))
        self.display.blit(gas_label, (label_x, gas_y + 2))

        gas_bar_bg = pygame.Rect(bar_x, gas_y, bar_width, bar_height)
        gas_bar_fill = pygame.Rect(bar_x, gas_y, int(bar_width * self.control.throttle), bar_height)
        pygame.draw.rect(self.display, (40, 40, 40), gas_bar_bg)
        pygame.draw.rect(self.display, (0, 200, 0), gas_bar_fill)
        pygame.draw.rect(self.display, (255, 255, 255), gas_bar_bg, 2)

        # Prozentanzeige für Gas
        gas_percent = font_small.render(f"{self.control.throttle * 100:.0f}%", True, (255, 255, 255))
        self.display.blit(gas_percent, (bar_x + bar_width + 10, gas_y + 2))

        # Brems-Anzeige (rot)
        brake_y = gas_y + bar_spacing
        brake_label = font_small.render("Bremse:", True, (255, 255, 255))
        self.display.blit(brake_label, (label_x, brake_y + 2))

        brake_bar_bg = pygame.Rect(bar_x, brake_y, bar_width, bar_height)
        brake_bar_fill = pygame.Rect(bar_x, brake_y, int(bar_width * self.control.brake), bar_height)
        pygame.draw.rect(self.display, (40, 40, 40), brake_bar_bg)
        pygame.draw.rect(self.display, (220, 0, 0), brake_bar_fill)
        pygame.draw.rect(self.display, (255, 255, 255), brake_bar_bg, 2)

        # Prozentanzeige für Bremse
        brake_percent = font_small.render(f"{self.control.brake * 100:.0f}%", True, (255, 255, 255))
        self.display.blit(brake_percent, (bar_x + bar_width + 10, brake_y + 2))

        # Lenk-Anzeige (blau, zentriert)
        steer_y = brake_y + bar_spacing
        steer_label = font_small.render("Lenkung:", True, (255, 255, 255))
        self.display.blit(steer_label, (label_x, steer_y + 2))

        steer_bar_bg = pygame.Rect(bar_x, steer_y, bar_width, bar_height)
        steer_center = bar_x + bar_width // 2  # Mitte der Lenkungsanzeige
        steer_width = int((bar_width // 2) * abs(self.control.steer))  # Breite basierend auf Lenkwinkel

        pygame.draw.rect(self.display, (40, 40, 40), steer_bar_bg)

        if self.control.steer < 0:  # Links lenken
            steer_bar_fill = pygame.Rect(steer_center - steer_width, steer_y, steer_width, bar_height)
        else:  # Rechts lenken
            steer_bar_fill = pygame.Rect(steer_center, steer_y, steer_width, bar_height)

        pygame.draw.rect(self.display, (0, 150, 255), steer_bar_fill)
        pygame.draw.rect(self.display, (255, 255, 255), steer_bar_bg, 2)

        # Mittellinie für Lenkung
        pygame.draw.line(self.display, (255, 255, 255),
                         (steer_center, steer_y),
                         (steer_center, steer_y + bar_height), 2)

        # Lenkwinkel in Prozent
        steer_percent = font_small.render(f"{self.control.steer * 100:.0f}%", True, (255, 255, 255))
        self.display.blit(steer_percent, (bar_x + bar_width + 10, steer_y + 2))

        # Handbremse-Indikator (oben über dem Tacho)
        if self.control.hand_brake:
            handbrake_text = font_medium.render("🚫 HANDBREMSE", True, (255, 50, 50))
            handbrake_rect = handbrake_text.get_rect(center=(tacho_x + tacho_width // 2, tacho_y - 25))

            # Hintergrund für Handbremse-Warnung
            warning_bg = pygame.Rect(handbrake_rect.x - 10, handbrake_rect.y - 5,
                                     handbrake_rect.width + 20, handbrake_rect.height + 10)
            pygame.draw.rect(self.display, (100, 0, 0), warning_bg)
            pygame.draw.rect(self.display, (255, 50, 50), warning_bg, 2)

            self.display.blit(handbrake_text, handbrake_rect)

    def render_info(self):
        """Benchmark-Info auf dem Display anzeigen"""
        if not hasattr(pygame, 'font') or not pygame.font.get_init():
            return

        font = pygame.font.Font(None, 36)
        info_text = []

        if self.benchmark_active:
            current_time = time.time() - self.start_time
            info_text.append(f"Zeit: {current_time:.1f}s")
            if self.deviations:
                current_deviation = self.deviations[-1]
                mean_deviation = np.mean(self.deviations)
                info_text.append(f"Aktuelle Abweichung: {current_deviation:.2f}m")
                info_text.append(f"Mittlere Abweichung: {mean_deviation:.2f}m")
            info_text.append(f"Waypoint: {self.last_wp_idx}/{len(self.route) if self.route else 0}")
        else:
            info_text.append("Drücke 'S' zum Starten")
            info_text.append("Drücke 'R' zum Zurücksetzen")
            info_text.append("Pfeiltasten: Fahrzeugsteuerung")

        # Text rendern
        y_offset = 10
        for text in info_text:
            text_surface = font.render(text, True, (255, 255, 255))
            # Schwarzer Hintergrund für bessere Lesbarkeit
            text_rect = text_surface.get_rect()
            bg_rect = pygame.Rect(10, y_offset, text_rect.width + 10, text_rect.height + 5)
            pygame.draw.rect(self.display, (0, 0, 0), bg_rect)
            self.display.blit(text_surface, (15, y_offset + 2))
            y_offset += text_rect.height + 10

    def cleanup(self):
        """Ressourcen aufräumen"""
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        if self.display:
            pygame.quit()

    def run(self):

        self.setup_pygame()

        try:
            running = True
            while running:
                # Events verarbeiten
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_s and not self.benchmark_active:
                            self.start_benchmark()
                        elif event.key == pygame.K_r:
                            # Route zurücksetzen - Fahrzeug zur Startposition
                            self.reset_vehicle_to_start()

                # Fahrzeug-Input verarbeiten
                if self.vehicle:
                    self.handle_input()

                # Benchmark aktualisieren
                self.update_benchmark()

                # Display aktualisieren
                if not self.camera:  # Falls keine Kamera, schwarzer Bildschirm
                    self.display.fill((0, 0, 0))

                self.render_info()
                self.render_speedometer()  # Tacho hinzufügen
                pygame.display.flip()
                self.clock.tick(60)

        finally:
            self.cleanup()


def main():
    # Beispiel-Nutzung
    benchmark = CarlaBenchmark()

    # Hier würdest du deine Route setzen
    # Beispiel (du musst diese durch deine echten Daten ersetzen):

    # spawn_point = carla.Transform(carla.Location(x=100, y=200, z=1))
    # waypoint_list = [...]  # Deine Waypoint-Liste
    # benchmark.set_route(spawn_point, waypoint_list)
    benchmark.spawn_vehicle(benchmark.spawn_point)
    benchmark.run()

    # benchmark.run()

    return benchmark


if __name__ == "__main__":
    main()