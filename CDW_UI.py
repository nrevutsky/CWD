from kivy.app import App
from kivy.uix.scatter import Scatter
from kivy.uix.label import Label
from kivy.uix.scatter import ScatterPlane
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.properties import StringProperty
from kivy.properties import NumericProperty
from kivy.uix.image import AsyncImage
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.graphics import Rectangle
from kivy.core.image import Image
from kivy.properties import ObjectProperty
from kivy.properties import NumericProperty
from kivy.graphics.transformation import Matrix
from kivy.graphics.context_instructions import *
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.slider import Slider
from kivy.cache import Cache
import numpy
from Open import Dataset
import os


class GridMapOverview(GridLayout):
    A_image = ObjectProperty(None)

    def __init__(self, **kw):
        super(GridMapOverview, self).__init__(**kw)


class ScrLayot(ScatterPlane):
    def __init__(self, **kwargs):
        super(ScrLayot, self).__init__(**kw)


class AreaLevel(Widget):
    def __init__(self, **kw):
        super(AreaLevel, self).__init__(**kw)


class ZoomBox(BoxLayout):
    def __init__(self, **kw):
        super(ZoomBox, self).__init__(**kw)


class OpenWindow(GridLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    ext_input = ObjectProperty(None)
    A_image = ObjectProperty(None)
    a = ObjectProperty(None)
    Dat = ObjectProperty(None)
    f_name = StringProperty("TMP.tiff")
    Band_num = NumericProperty(0)

    def __init__(self, **kw):
        super(OpenWindow, self).__init__(**kw)
        self.cols = 2

    def dismiss_popup(self):
        self._popup.dismiss()
        self.cols

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))

        self._popup.open()

    def load(self, path, filename):
        self.Dat = (Dataset(''.join(filename)))
        self.Dat.SaveBAND(self.Dat.BandsArrays[self.Band_num], "TMP.tiff")
        self.A_image = Image("TMP.tiff")

        self.dismiss_popup()

    def save(self, path, filename):

        self.dismiss_popup()

    def next(self):
        if self.Band_num < len(self.Dat.BandsArrays) - 1:
            self.Band_num += 1
        Cache.remove('kv.image')
        Cache.remove('kv.texture')
        self.Dat.SaveBAND(self.Dat.BandsArrays[self.Band_num], "TMP.tiff")
        self.A_image = Image("TMP.tiff")

    def previous(self):
        if self.Band_num >= 0:
            self.Band_num -= 1
        Cache.remove('kv.image')
        Cache.remove('kv.texture')
        self.Dat.SaveBAND(self.Dat.BandsArrays[self.Band_num], "TMP.tiff")
        self.A_image = Image("TMP.tiff")

    def Gistogram(self):
        self.Dat.SaveBAND(self.Dat.histequalization(self.Dat.ListofBands[self.Band_num]), "TMP.tiff")
        Cache.remove('kv.image')
        Cache.remove('kv.texture')
        self.A_image = Image("TMP.tiff")

    def Composite(self):
        self.Dat.CompositeBands(self.Dat.BandsArrays[5], self.Dat.BandsArrays[6], self.Dat.BandsArrays[3],
                                self.Dat.BandsArrays[2])
        Cache.remove('kv.image')
        Cache.remove('kv.texture')
        self.A_image = Image("TMP.tiff")

    def I_NDVI(self):
        self.Dat.NDVI(self.Dat.BandsArrays[4], self.Dat.BandsArrays[7])
        Cache.remove('kv.image')
        Cache.remove('kv.texture')
        self.A_image = Image("TMP.tiff")

    def I_IOR(self):
        self.Dat.IOR(self.Dat.BandsArrays[5], self.Dat.BandsArrays[3], self.Dat.BandsArrays[4])
        Cache.remove('kv.image')
        Cache.remove('kv.texture')
        self.A_image = Image("TMP.tiff")

    def PanSharp(self):
        self.Dat.BandsArrays = self.Dat.pansharpen('P002.TIF')


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    raster = ObjectProperty(None)
    cancel = ObjectProperty(None)


class CDWApp(App):
    def build(self):
        # self.load_kv('cdw.kv')
        return OpenWindow()


Factory.register('OpenWindow', cls=OpenWindow)
Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)

if __name__ == "__main__":
    CDWApp().run()
