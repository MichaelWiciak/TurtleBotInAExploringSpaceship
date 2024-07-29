from group_project.astro_stitch import astro_stitch, show_image
import cv2

view_earth = cv2.imread("./astro_stitch_test_files/viewEarth.jpg")
view_moon = cv2.imread("./astro_stitch_test_files/viewMoon.jpg")


def main():
    try:
        stitched = astro_stitch(view_earth, view_moon)

        show_image(stitched)
        pass
    except Exception as e:
        print(e)
        import traceback

        print(traceback.format_exc())
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
