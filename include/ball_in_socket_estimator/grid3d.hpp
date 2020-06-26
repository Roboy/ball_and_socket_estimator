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
    unsigned theta_steps, phi_steps;
    vector<vector<Vec3<T>>> data;

    Grid(unsigned theta_steps, unsigned phi_steps, vector<vector<int>> indices, vector<vector<vector<double>>> values) :
      theta_steps(theta_steps),phi_steps(phi_steps) {
        data.resize(theta_steps);
        for (int x = 0; x < theta_steps; x++) {
          data[x].resize(phi_steps);
          for (int y = 0; y < phi_steps; y++) {
                int index = indices[x][y];
                data[x][y] = Vec3<T>(values[index][x][0],values[index][x][1],values[index][x][2]);
            }
        }
    }

    ~Grid() {}

    Vec3<T> interpolate(float x, float y) {
        T gx, gy, tx, ty;
        unsigned gxi, gyi;
        // remap point coordinates to grid coordinates
        gx = x * theta_steps;
        gxi = int(gx);
        tx = gx - gxi;
        gy = y * phi_steps;
        gyi = int(gy);
        ty = gy - gyi;
        if(gxi>=theta_steps-1)
          gxi = theta_steps-2;
        if(gyi>=phi_steps-1)
          gyi = phi_steps-2;
        const Vec3<T> &c000 = data[gxi][gyi];
        const Vec3<T> &c100 = data[gxi + 1][gyi];
        const Vec3<T> &c010 = data[gxi][gyi + 1];
        const Vec3<T> &c110 = data[gxi + 1][gyi + 1];
        return
                (T(1) - tx) * (T(1) - ty)  * c000 +
                tx * (T(1) - ty) * c100 +
                (T(1) - tx) * ty * c010 +
                tx * ty * c110;
    }
};
