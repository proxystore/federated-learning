from dataclasses import dataclass
from proxystore.connectors.redis import RedisConnector
from proxystore.store import get_store, register_store, Store


if __name__ == '__main__':
    @dataclass
    class Person:
        name: str
        age: int

        def legal_to_drink(self) -> bool:
            return self.age >= 21

    store = Store('my-store', RedisConnector(hostname='localhost', port=1234))
    register_store(store)
    store = get_store('my-store')

    person_obj = Person('Bob Ross', 45)
    key = store.set(person_obj)
    assert person_obj == store.get(key)

    prx = store.proxy(person_obj)

    assert isinstance(prx, type(person_obj))
    print(f'Can our person object (via proxy), by the name of {prx.name}, legally drink? [{prx.legal_to_drink()}]')
