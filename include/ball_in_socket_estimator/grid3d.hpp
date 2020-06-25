#include <vector>

template<typename T>
class Vec3 {
public:
    Vec3() : x(T(0)), y(T(0)), z(T(0)) {}

    Vec3(T xx) : x(xx), y(xx), z(xx) {}

    Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}

    Vec3 operator+(const Vec3 &v) const { return Vec3(x + v.x, y + v.y, z + v.z); }

    Vec3 operator-(const Vec3 &v) const { return Vec3(x - v.x, y - v.y, z - v.z); }

    Vec3 operator-() const { return Vec3(-x, -y, -z); }

    Vec3 operator*(const T &r) const { return Vec3(x * r, y * r, z * r); }

    Vec3 operator*(const Vec3 &v) const { return Vec3(x * v.x, y * v.y, z * v.z); }

    T dotProduct(const Vec3<T> &v) const { return x * v.x + y * v.y + z * v.z; }

    Vec3 &operator/=(const T &r) {
        x /= r, y /= r, z /= r;
        return *this;
    }

    Vec3 &operator*=(const T &r) {
        x *= r, y *= r, z *= r;
        return *this;
    }

    Vec3 crossProduct(const Vec3<T> &v) const {
        return Vec3<T>(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    T norm() const { return x * x + y * y + z * z; }

    T length() const
    { return sqrt(norm()); }

    const T &operator[](uint8_t i) const { return (&x)[i]; }

    T &operator[](uint8_t i) { return (&x)[i]; }

    Vec3 &normalize() {
        T n = norm();
        if (n > 0) {
            T factor = 1 / sqrt(n);
            x *= factor, y *= factor, z *= factor;
        }

        return *this;
    }

    friend Vec3 operator*(const T &r, const Vec3 &v) { return Vec3<T>(v.x * r, v.y * r, v.z * r); }

    friend Vec3 operator/(const T &r, const Vec3 &v) { return Vec3<T>(r / v.x, r / v.y, r / v.z); }

    friend std::ostream &operator<<(std::ostream &s, const Vec3<T> &v) {
        return s << '[' << v.x << ' ' << v.y << ' ' << v.z << ']';
    }

    T x, y, z;
};

typedef Vec3<float> Vec3f;

float randUniform() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}
using namespace std;

template<typename T>
class Grid {
public:
    // unsigned nvoxels; // number of voxels (cube)
    // unsigned nx, ny, nz; // number of vertices
    unsigned nvoxels;
    unsigned x_steps, y_steps;
    Vec3<T> *data;

    Grid(unsigned x_steps, unsigned y_steps, vector<vector<int>> indices, vector<vector<vector<double>>> values) :
      x_steps(x_steps),y_steps(y_steps),data(nullptr) {
        data = new Vec3<T>[x_steps*y_steps];
        for (int x = 0; x < x_steps; x++) {
          for (int y = 0; y < y_steps; y++) {
                int index = indices[x][y];
                data[IX(x, y)] = Vec3<T>(values[index][x][0],values[index][x][1],values[index][x][2]);
            }
        }
    }

    ~Grid() { if (data) delete[] data; }

    unsigned IX(unsigned x, unsigned y) {
        if ((x > x_steps)) x -= 1;
        if ((y > y_steps)) y -= 1;
        return x * y_steps + y;
    }

    Vec3<T> interpolate(float x, float y) {
        T gx, gy, tx, ty;
        unsigned gxi, gyi;
        // remap point coordinates to grid coordinates
        gx = x * x_steps;
        gxi = int(gx);
        tx = gx - gxi;
        gy = y * y_steps;
        gyi = int(gy);
        ty = gy - gyi;
        const Vec3<T> &c000 = data[IX(gxi, gyi)];
        const Vec3<T> &c100 = data[IX(gxi + 1, gyi)];
        const Vec3<T> &c010 = data[IX(gxi, gyi + 1)];
        const Vec3<T> &c110 = data[IX(gxi + 1, gyi + 1)];
        return
                (T(1) - tx) * (T(1) - ty)  * c000 +
                tx * (T(1) - ty) * c100 +
                (T(1) - tx) * ty * c010 +
                tx * ty * c110;
    }
};
