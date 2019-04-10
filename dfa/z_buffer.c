// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>


int cover_point(double pt_x, double pt_y, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double pt3_x, double pt3_y)
{
    double v0_x = pt3_x - pt1_x;
    double v0_y = pt3_y - pt1_y;
    double v1_x = pt2_x - pt1_x;
    double v1_y = pt2_y - pt1_y;
    double v2_x = pt_x - pt1_x;
    double v2_y = pt_y - pt1_y;

    double dot00 = v0_x * v0_x + v0_y * v0_y;
    double dot01 = v0_x * v1_x + v0_y * v1_y;
    double dot02 = v0_x * v2_x + v0_y * v2_y;
    double dot11 = v1_x * v1_x + v1_y * v1_y;
    double dot12 = v1_x * v2_x + v1_y * v2_y;

    double inv = 1.0 / (dot00 * dot11 - dot01 * dot01);
    double u = (dot11 * dot02 - dot01 * dot12) * inv;
    if (u < 0 || u > 1) return 0;
    double v = (dot00 * dot12 - dot01 * dot02) * inv;
    if (v < 0 || v > 1) return 0;
    return (u + v <= 1);
}


static PyObject* z_buffer_c(PyObject* self, PyObject* args)
{
    int n, h, w, i;
    PyObject *arg_pt1 = NULL, *arg_pt2 = NULL, *arg_pt3 = NULL, *arg_r = NULL, *arg_dis = NULL, *arg_idx = NULL;
    PyArrayObject *arr_pt1 = NULL, *arr_pt2 = NULL, *arr_pt3 = NULL, *arr_r = NULL, *arr_dis = NULL, *arr_idx = NULL;

    if (!PyArg_ParseTuple(args, "iiiOOOOO!O!", &n, &h, &w, &arg_pt1, &arg_pt2, &arg_pt3, &arg_r, &PyArray_Type, &arg_dis, &PyArray_Type, &arg_idx)) return NULL;

    arr_pt1 = PyArray_FROM_OTF(arg_pt1, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr_pt1 == NULL) goto fail;
    arr_pt2 = PyArray_FROM_OTF(arg_pt2, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr_pt2 == NULL) goto fail;
    arr_pt3 = PyArray_FROM_OTF(arg_pt3, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr_pt3 == NULL) goto fail;
    arr_r = PyArray_FROM_OTF(arg_r, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr_r == NULL) goto fail;
    arr_dis = PyArray_FROM_OTF(arg_dis, NPY_DOUBLE, NPY_INOUT_ARRAY);
    if (arr_dis == NULL) goto fail;
    arr_idx = PyArray_FROM_OTF(arg_idx, NPY_INT32, NPY_INOUT_ARRAY);
    if (arr_idx == NULL) goto fail;

    for (i = 0; i < n; i++) {
        double *pt1_x = (double *)PyArray_GETPTR2(arr_pt1, 0, i);
        double *pt1_y = (double *)PyArray_GETPTR2(arr_pt1, 1, i);
        double *pt2_x = (double *)PyArray_GETPTR2(arr_pt2, 0, i);
        double *pt2_y = (double *)PyArray_GETPTR2(arr_pt2, 1, i);
        double *pt3_x = (double *)PyArray_GETPTR2(arr_pt3, 0, i);
        double *pt3_y = (double *)PyArray_GETPTR2(arr_pt3, 1, i);

        int umin = ceil(fmin(fmin(*pt1_x, *pt2_x), *pt3_x));
        int umax = floor(fmax(fmax(*pt1_x, *pt2_x), *pt3_x));
        int vmin = ceil(fmin(fmin(*pt1_y, *pt2_y), *pt3_y));
        int vmax = floor(fmax(fmax(*pt1_y, *pt2_y), *pt3_y));

        if (umax < umin || vmax < vmin || umax > w || umin < 1 || vmax > h || vmin < 1) continue;

        int u, v, *idx;
        double *dis, *r;
        for (u = umin - 1; u < umax; u++) {
            for (v = vmin - 1; v < vmax; v++) {
                dis = (double *)PyArray_GETPTR2(arr_dis, v, u);
                idx = (int *)PyArray_GETPTR2(arr_idx, v, u);
                r = (double *)PyArray_GETPTR1(arr_r, i);
                if (*dis < *r && cover_point(u + 1, v + 1, *pt1_x, *pt1_y, *pt2_x, *pt2_y, *pt3_x, *pt3_y)) {
                    *dis = *r;
                    *idx = i + 1;
                }
            }
        }
    }

    Py_DECREF(arr_pt1);
    Py_DECREF(arr_pt2);
    Py_DECREF(arr_pt3);
    Py_DECREF(arr_r);
    Py_DECREF(arr_dis);
    Py_DECREF(arr_idx);
    Py_INCREF(Py_None);
    return Py_None;

fail:
    Py_XDECREF(arr_pt1);
    Py_XDECREF(arr_pt2);
    Py_XDECREF(arr_pt3);
    Py_XDECREF(arr_r);
    PyArray_XDECREF_ERR(arr_dis);
    PyArray_XDECREF_ERR(arr_idx);
    return NULL;
}


static PyMethodDef Methods[] =
{
    {"z_buffer_c", z_buffer_c, METH_VARARGS, "compute the Z buffer"},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC
initz_buffer(void)
{
    (void)Py_InitModule("z_buffer", Methods);
    import_array();
}