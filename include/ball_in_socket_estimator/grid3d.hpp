

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

template<typename T>
class Grid {
public:
    unsigned nvoxels; // number of voxels (cube)
    unsigned nx, ny, nz; // number of vertices
    Vec3<T> *data;

    Grid(unsigned nv = 10) : nvoxels(nv), data(nullptr) {
        nx = ny = nz = nvoxels + 1;
        data = new Vec3<T>[nx * ny * nz];
        for (int z = 0; z < nz + 1; ++z) {
            for (int y = 0; y < ny + 1; ++y) {
                for (int x = 0; x < nx + 1; ++x) {
                    data[IX(x, y, z)] = Vec3<T>(randUniform(), randUniform(), randUniform());
                }
            }
        }
    }

    ~Grid() { if (data) delete[] data; }

    unsigned IX(unsigned x, unsigned y, unsigned z) {
        if (!(x < nx)) x -= 1;
        if (!(y < ny)) y -= 1;
        if (!(z < nz)) z -= 1;
        return x * nx * ny + y * nx + z;
    }

    Vec3<T> interpolate(const Vec3<T> &location) {
        T gx, gy, gz, tx, ty, tz;
        unsigned gxi, gyi, gzi;
        // remap point coordinates to grid coordinates
        gx = location.x * nvoxels;
        gxi = int(gx);
        tx = gx - gxi;
        gy = location.y * nvoxels;
        gyi = int(gy);
        ty = gy - gyi;
        gz = location.z * nvoxels;
        gzi = int(gz);
        tz = gz - gzi;
        const Vec3<T> &c000 = data[IX(gxi, gyi, gzi)];
        const Vec3<T> &c100 = data[IX(gxi + 1, gyi, gzi)];
        const Vec3<T> &c010 = data[IX(gxi, gyi + 1, gzi)];
        const Vec3<T> &c110 = data[IX(gxi + 1, gyi + 1, gzi)];
        const Vec3<T> &c001 = data[IX(gxi, gyi, gzi + 1)];
        const Vec3<T> &c101 = data[IX(gxi + 1, gyi, gzi + 1)];
        const Vec3<T> &c011 = data[IX(gxi, gyi + 1, gzi + 1)];
        const Vec3<T> &c111 = data[IX(gxi + 1, gyi + 1, gzi + 1)];
        return
                (T(1) - tx) * (T(1) - ty) * (T(1) - tz) * c000 +
                tx * (T(1) - ty) * (T(1) - tz) * c100 +
                (T(1) - tx) * ty * (T(1) - tz) * c010 +
                tx * ty * (T(1) - tz) * c110 +
                (T(1) - tx) * (T(1) - ty) * tz * c001 +
                tx * (T(1) - ty) * tz * c101 +
                (T(1) - tx) * ty * tz * c011 +
                tx * ty * tz * c111;
    }
};
