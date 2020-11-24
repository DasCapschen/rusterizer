/*!
    Implementation of a (probably naïve) Entity Component System
*/

use std::{any::Any, sync::RwLock, collections::{HashSet, HashMap}, sync::Arc};
use std::any::TypeId;
use std::collections::hash_map::Keys;

// TODO: some kind of execution-order for systems?

struct ECSSystems(pub HashMap<TypeId, Box<dyn System>>);
impl ECSSystems {
    fn run_all(&self, components: &mut ECSComponents) {
        for system in self.0.values() {
            system.run(components);
        }
    }
}

struct ECSComponents(HashMap<TypeId, HashMap<Entity, Arc<dyn Any + Send + Sync + 'static>>>);

//this is just... wow
struct ECSWorld {
    systems: ECSSystems,
    components: ECSComponents
}

impl ECSWorld {
    fn register_system<T: System>(&mut self) {
        let typeid = TypeId::of::<T>();
        if !self.systems.contains_key(&typeid) {
            self.systems.insert(typeid, Box::new(T::new()));
        }
    }

    fn run_all_systems(&mut self) {
        self.systems.0.values().for_each(|system| system.run(&mut self.components));
    }

    //i feel like maybe we should have an error that isn't just a string...
    fn add_component<T: Component>(&mut self, entity: Entity, component: T) -> Result<(), &str> {
        let typeid = TypeId::of::<T>();
        let entry = self.components.entry(typeid).or_insert(Default::default());
        if entry.contains_key(&entity) {
            Err("Entity already has this component. No changes made.")
        } else {
            entry.insert(entity, Arc::new(RwLock::new(component)));
            Ok(())
        }
    }

    fn get_component<T: Component>(&self, entity: Entity) -> Option<Arc<RwLock<T>>> {
        let typeid = TypeId::of::<T>();
        if let Some(comp) = self.components.get(&typeid).and_then(|m| m.get(&entity)) {
            Some( comp.clone().downcast::<RwLock<T>>().expect("Couldn't downcast. RwLock missing?") )
        } else {
            None
        }
    }
}

trait System : 'static {
    fn new() -> Self where Self: Sized;
    fn run(&self, world: &mut ECSWorld);
}
trait Component : 'static + Send + Sync { }

struct TestComp{
    data: i32
}
impl Component for TestComp {}

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone)]
struct Entity(usize);

struct TestSystem {}
impl System for TestSystem {
    fn new() -> Self where Self: Sized {
        TestSystem {}
    }

    fn run(&self, world: &mut ECSWorld) {
        let entity = Entity(1);

        world.add_component(entity, TestComp{ data: 8 });

        let comp = world.get_component::<TestComp>(entity).unwrap();



    }
}