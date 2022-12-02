from PIL import Image
from django.http import Http404
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

from utils.mat_test import test_mat


# Create your views here.

@api_view(['POST'])
def test(request):
    mat1 = request.FILES.get('mat1')
    mat2 = request.FILES.get('mat2')
    source_1_path = 'media/source' + ''.join(str(mat1).split('.')[:-1]) + '.mat'
    source_2_path = 'media/source' + ''.join(str(mat2).split('.')[:-1]) + '.mat'
    with open(source_1_path, 'wb+') as f:
        for chunk in mat1.chunks():
            f.write(chunk)
    with open(source_2_path, 'wb+') as f:
        for chunk in mat2.chunks():
            f.write(chunk)
    result_path = 'media/result' + ''.join(str(mat1).split('.')[:-1]) + '.png'
    result = test_mat(source_1_path, source_2_path, result_path)
    if result:
        return Response({
            'result': 'http://' + request.get_host() + '/' + result_path
        })
    else:
        raise Http404


def index(request):
    return render(request, 'main.html')
