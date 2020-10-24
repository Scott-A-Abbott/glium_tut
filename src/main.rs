#[macro_use]
extern crate glium;
use {
    fnv::FnvHashMap,
    glium::{
        glutin::{
            event::{Event, StartCause, WindowEvent},
            event_loop::{ControlFlow, EventLoop},
            window::WindowBuilder,
            ContextBuilder,
        },
        index::{NoIndices, PrimitiveType},
        texture::{RawImage2d, SrgbTexture2d, Texture2d},
        Display, IndexBuffer, Program, Surface, VertexBuffer,
    },
    std::{
        io::Cursor,
        marker::PhantomData,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, RwLock,
        },
        time::{Duration, Instant},
    },
};
mod teapot;

#[derive(Eq, Hash, PartialEq, Debug)]
pub struct Handle<T: ?Sized> {
    id: Arc<usize>,
    marker: PhantomData<T>,
}
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Handle {
            id: self.id.clone(),
            marker: PhantomData,
        }
    }
}

#[derive(Default)]
pub struct AssetStorage<T> {
    assets: FnvHashMap<usize, T>,
    index: AtomicUsize,
    // handles: Vec<Handle<T>>,
}
impl<T> AssetStorage<T> {
    pub fn new() -> Self {
        Self {
            assets: Default::default(),
            index: Default::default(),
            // handles: Default::default(),
        }
    }

    pub fn allocate(&mut self) -> Handle<T> {
        let index = self.index.fetch_add(1, Ordering::Relaxed);
        Handle {
            id: Arc::new(index),
            marker: PhantomData,
        }
    }

    pub fn insert(&mut self, asset: T, handle: Handle<T>) {
        self.assets.insert(*handle.id, asset);
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        self.assets.get(&handle.id)
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        self.assets.get_mut(&handle.id)
    }

    pub fn contains(&self, handle: Handle<T>) -> bool {
        self.assets.contains_key(&handle.id)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Status {
    Loading,
    Complete,
}
#[derive(Default, Debug)]
pub struct ProgressCounter {
    num_assets: usize,
    num_loading: Arc<AtomicUsize>,
}
impl ProgressCounter {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn add_asset(&mut self) -> ProgressTracker {
        self.num_assets += 1;
        self.num_loading.fetch_add(1, Ordering::Relaxed);

        ProgressTracker::new(self.num_loading.clone())
    }

    pub fn status(&self) -> Status {
        match self.num_loading.load(Ordering::Relaxed) {
            0 => Status::Complete,
            _ => Status::Loading,
        }
    }

    pub fn loaded(&self) -> usize {
        self.num_assets - self.num_loading.load(Ordering::Relaxed)
    }

    pub fn assets(&self) -> usize {
        self.num_assets
    }
}
pub struct ProgressTracker {
    num_loading: Arc<AtomicUsize>,
}
impl ProgressTracker {
    fn new(num_loading: Arc<AtomicUsize>) -> Self {
        Self { num_loading }
    }
    pub fn mark_complete(&mut self) {
        self.num_loading.fetch_sub(1, Ordering::Relaxed);
    }
}

type SafeStorage<T> = Arc<RwLock<AssetStorage<T>>>;
pub fn create_storage<T>() -> SafeStorage<T> {
    Arc::new(RwLock::new(AssetStorage::<T>::new()))
}

pub struct Loader;
impl Loader {
    pub fn load<T, F>(progress: &mut ProgressCounter, storage: SafeStorage<T>, f: F) -> Handle<T>
    where
        F: FnOnce() -> T + Send + Sync + 'static,
        T: Send + Sync + 'static,
    {
        let handle = storage.write().unwrap().allocate();
        let mut tracker = progress.add_asset();
        let handle_clone = handle.clone();
        rayon::spawn(move || {
            let asset = f();
            storage.write().unwrap().insert(asset, handle_clone);
            tracker.mark_complete();
        });

        handle
    }
}

fn main() {
    #[derive(Copy, Clone)]
    struct Vertex {
        position: [f32; 3],
        normal: [f32; 3],
        tex_coords: [f32; 2],
    }
    implement_vertex!(Vertex, position, normal, tex_coords);
    let width = 1.57 / 1.85;

    let shape = [
        Vertex {
            position: [-width, 1.0, 0.0],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [0.0, 1.0],
        },
        Vertex {
            position: [width, 1.0, 0.0],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [1.0, 1.0],
        },
        Vertex {
            position: [-width, -1.0, 0.0],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [0.0, 0.0],
        },
        Vertex {
            position: [width, -1.0, 0.0],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [1.0, 0.0],
        },
    ];

    let event_loop = EventLoop::new();
    let wb = WindowBuilder::new();
    let cb = ContextBuilder::new().with_depth_buffer(24);
    let display = Display::new(wb, cb, &event_loop).unwrap();
    let mut progress_counter = ProgressCounter::new();
    let image_storage = create_storage::<image::RgbaImage>();

    let vb = VertexBuffer::new(&display, &shape).unwrap();

    let _positions = VertexBuffer::new(&display, &teapot::VERTICES).unwrap();
    let _normals = VertexBuffer::new(&display, &teapot::NORMALS).unwrap();
    let _indices =
        IndexBuffer::new(&display, PrimitiveType::TrianglesList, &teapot::INDICES).unwrap();

    let diffuse_handle = Loader::load(&mut progress_counter, image_storage.clone(), || {
        image::load(
            Cursor::new(&include_bytes!("../_laigter_test.png")[..]),
            image::ImageFormat::Png,
        )
        .unwrap()
        .to_rgba()
    });

    let normal_map_handle = Loader::load(&mut progress_counter, image_storage.clone(), || {
        image::load(
            Cursor::new(&include_bytes!("../_laigter_test_n.png")[..]),
            image::ImageFormat::Png,
        )
        .unwrap()
        .to_rgba()
    });

    // let _texture = Texture2d::new(&display, image).unwrap();

    let vertex_shader = r#"
        #version 330 core
        
        in vec3 position;
        in vec3 normal;
        in vec2 tex_coords;
        
        out vec2 v_tex_coords;
        out vec3 v_normal;
        out vec3 v_position;

        uniform mat4 view;
        uniform mat4 model;
        uniform mat4 perspective;

        void main() {
            v_tex_coords = tex_coords;
            mat4 modelview = view * model;
            v_normal = transpose(inverse(mat3(model))) * normal;
            gl_Position = perspective * modelview * vec4(position, 1.0);
            v_position = gl_Position.xyz / gl_Position.w;
        }
    "#;

    let fragment_shader = r#"
        #version 330 core

        in vec2 v_tex_coords;
        in vec3 v_normal;
        in vec3 v_position;
        out vec4 color;

        uniform vec3 u_light;
        uniform sampler2D diffuse_tex;
        uniform sampler2D normal_tex;

        const vec4 specular_color = vec4(1.0, 1.0, 1.0, 1.0);

        mat3 cotangent_frame(vec3 normal, vec3 pos, vec2 uv) {
            vec3 dp1 = dFdx(pos);
            vec3 dp2 = dFdy(pos);
            vec2 duv1 = dFdx(uv);
            vec2 duv2 = dFdy(uv);
        
            vec3 dp2perp = cross(dp2, normal);
            vec3 dp1perp = cross(normal, dp1);
            vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
            vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
        
            float invmax = inversesqrt(max(dot(T, T), dot(B, B)));
            return mat3(T * invmax, B * invmax, normal);
        }

        void main() {
            vec4 diffuse_color = texture(diffuse_tex, v_tex_coords);
            vec4 ambient_color = vec4(diffuse_color.rgb * 0.1, diffuse_color.a);

            vec3 normal_map = texture(normal_tex, v_tex_coords).rgb;
            mat3 tbn = cotangent_frame(v_normal, v_position, v_tex_coords);
            vec3 real_normal = normalize(tbn * -(normal_map * 2.0 - 1.0));

            float diffuse = max(dot(real_normal, normalize(u_light)), 0.0);

            vec3 camera_dir = normalize(-v_position);
            vec3 half_direction = normalize(normalize(u_light) + camera_dir);
            float specular = pow(max(dot(half_direction, real_normal), 0.0), 16.0);

            color = vec4(ambient_color + diffuse * diffuse_color + specular * specular_color);
        }
    "#;

    let program = Program::from_source(&display, vertex_shader, fragment_shader, None).unwrap();

    let mut t: f32 = -1.5;
    let light = [1.4, 0.4, -0.7f32];

    let mut diffuse_texture = None;
    let mut normal_map = None;

    event_loop.run(move |ev, _, control_flow| {
        let next_frame_time = Instant::now() + Duration::from_nanos(16_666_667);
        *control_flow = ControlFlow::WaitUntil(next_frame_time);

        t += 0.002;
        if t > 1.5 {
            t = -1.5;
        }
        match ev {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                    return;
                }
                _ => return,
            },
            Event::NewEvents(cause) => match cause {
                StartCause::ResumeTimeReached { .. } => (),
                StartCause::Init => (),
                _ => return,
            },
            _ => return,
        }

        let mut target = display.draw();
        target.clear_color_and_depth((0.392, 0.584, 0.929, 1.), 1.);

        let perspective = {
            let (w, h) = target.get_dimensions();
            let aspect_ratio = h as f32 / w as f32;
            let fov: f32 = 3.141592 / 3.0;
            let zfar = 1024.0;
            let znear = 0.1;
            let f = 1.0 / (fov / 2.).tan();

            [
                [f * aspect_ratio, 0., 0., 0.],
                [0., f, 0., 0.],
                [0., 0., (zfar + znear) / (zfar - znear), 1.],
                [0., 0., -(2. * zfar * znear) / (zfar - znear), 0.],
            ]
        };

        let view = view_matrix(&[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0], &[0.0, 1.0, 0.0]);
        let model = [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 2., 1.0f32],
        ];

        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                ..Default::default()
            },
            blend: glium::Blend::alpha_blending(),
            ..Default::default()
        };

        // println!(
        //     "Progress: {:?} {}/{}",
        //     progress_counter.status(),
        //     progress_counter.loaded(),
        //     progress_counter.assets()
        // );
        if progress_counter.status() == Status::Complete {
            match (&diffuse_texture, &normal_map) {
                (None, None) => {
                    diffuse_texture = {
                        let storage = image_storage.read().unwrap();
                        let image = storage.get(diffuse_handle.clone()).unwrap();
                        let image_dimensions = image.dimensions();
                        let image =
                            RawImage2d::from_raw_rgba_reversed(&image.to_vec(), image_dimensions);
                        let tex = SrgbTexture2d::new(&display, image).unwrap();
                        Some(tex)
                    };

                    normal_map = {
                        let storage = image_storage.read().unwrap();
                        let im = storage.get(normal_map_handle.clone()).unwrap();
                        let im_d = im.dimensions();
                        let im = RawImage2d::from_raw_rgba_reversed(&im.to_vec(), im_d);
                        let tex = Texture2d::new(&display, im).unwrap();
                        Some(tex)
                    };
                }
                (Some(diffuse_texture), Some(normal_map)) => {
                    let uni = uniform! {
                        model: model,
                        // tex: &texture,
                        perspective: perspective,
                        u_light: light,
                        view: view,
                        diffuse_tex: diffuse_texture,
                        normal_tex: normal_map
                    };

                    target
                        .draw(
                            &vb,
                            NoIndices(PrimitiveType::TriangleStrip),
                            &program,
                            &uni,
                            &params,
                        )
                        .unwrap();
                }
                (_, _) => {}
            }
        }

        //Draw here;
        // target
        //     .draw(&vb, &indices, &program, &uni, &Default::default())
        //     .unwrap();

        // target
        //     .draw((&positions, &normals), &indices, &program, &uni, &params)
        //     .unwrap();

        target.finish().unwrap();
    });
}

fn view_matrix(position: &[f32; 3], direction: &[f32; 3], up: &[f32; 3]) -> [[f32; 4]; 4] {
    let f = {
        let f = direction;
        let len = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
        let len = len.sqrt();
        [f[0] / len, f[1] / len, f[2] / len]
    };

    let s = [
        up[1] * f[2] - up[2] * f[1],
        up[2] * f[0] - up[0] * f[2],
        up[0] * f[1] - up[1] * f[0],
    ];

    let s_norm = {
        let len = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        let len = len.sqrt();
        [s[0] / len, s[1] / len, s[2] / len]
    };

    let u = [
        f[1] * s_norm[2] - f[2] * s_norm[1],
        f[2] * s_norm[0] - f[0] * s_norm[2],
        f[0] * s_norm[1] - f[1] * s_norm[0],
    ];

    let p = [
        -position[0] * s_norm[0] - position[1] * s_norm[1] - position[2] * s_norm[2],
        -position[0] * u[0] - position[1] * u[1] - position[2] * u[2],
        -position[0] * f[0] - position[1] * f[1] - position[2] * f[2],
    ];

    [
        [s_norm[0], u[0], f[0], 0.0],
        [s_norm[1], u[1], f[1], 0.0],
        [s_norm[2], u[2], f[2], 0.0],
        [p[0], p[1], p[2], 1.0],
    ]
}
